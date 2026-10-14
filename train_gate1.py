"""Gate 1 small trainable comparison. Reproducible but explicitly exploratory.

Examples:
    python train_gate1.py --smoke --device cpu
    python train_gate1.py --steps 1200 --seeds 41 97 127 --device auto
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from gate1.data import episodes
from gate1.models import (PhaseGraph, FixedRouterPhaseGraph, GRUMemory, FastWeightMemory,
                          LearnedWindowAttention, exact_window, exact_indexed)

MODELS = {'phase_graph': PhaseGraph, 'phase_fixed_router': FixedRouterPhaseGraph, 'gru': GRUMemory,
          'fast_weight': FastWeightMemory, 'window_attention': LearnedWindowAttention}
TRAIN_IDS = {'independent': 0, 'grouped': 1}
TEST_SCENARIOS = ('independent', 'grouped', 'broken_group', 'rewrite', 'grouped16')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_logits(model, batch):
    predictions, memory = model(batch.keys, batch.bits, batch.query)
    return predictions, memory


def loss_for(outputs, target, model_name):
    losses = {name: F.binary_cross_entropy_with_logits(logits, target.float())
              for name, logits in outputs.items()}
    if model_name in ('phase_graph', 'phase_fixed_router'):
        # Main gate: train both geometry and ordered probe; fixed-route reader
        # receives gradients of its own weights and is a negative control.
        return .42*losses['ratio'] + .20*losses['paired'] + .06*losses['single'] + .05*losses['fixed_route'] + .07*losses['learned_order_policy'] + .20*losses['direct_addressed']
    return next(iter(losses.values()))


def evaluate(models, device, *, seed=900000, cases=512, lengths=(16, 32, 64, 96)):
    rows = []
    with torch.no_grad():
        for scenario_index, scenario in enumerate(TEST_SCENARIOS):
            for n in lengths:
                generator = torch.Generator().manual_seed(seed + scenario_index*100003 + n*101)
                batch = episodes(cases, n, generator, device=device, scenario=scenario)
                for model_name, model in models.items():
                    model.eval()
                    outputs, state = model_logits(model, batch)
                    for reader_name, logit in outputs.items():
                        accuracy = ((logit >= 0).long() == batch.target).float().mean().item()
                        loss = F.binary_cross_entropy_with_logits(logit, batch.target.float()).item()
                        rows.append({'scenario': scenario, 'length': n,
                                     'model': model_name, 'readout': reader_name,
                                     'accuracy': round(accuracy, 6), 'bce': round(loss, 6),
                                     'episodes': cases})
                found, tape = exact_window(batch.keys, batch.bits, batch.query)
                # Unknown keys get expected 50%, not a magic correct answer.
                acc = torch.where(found, (tape == batch.target).float(), 0.5).mean().item()
                rows.append({'scenario':scenario, 'length':n, 'model':'exact_window_64B',
                             'readout':'last_match', 'accuracy':round(acc, 6), 'bce':None,
                             'episodes':cases, 'hit_rate':round(found.float().mean().item(),6)})
                exact = exact_indexed(batch.keys, batch.bits, batch.query)
                rows.append({'scenario':scenario,'length':n,'model':'indexed_bit_table_16B',
                             'readout':'exact','accuracy':round((exact==batch.target).float().mean().item(),6),
                             'bce':None,'episodes':cases})
    return rows


def main(args):
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    seed_receipts = []
    for seed in args.seeds:
        seed_everything(seed)
        models = {name: cls() for name, cls in MODELS.items()}
        # Matched initialization: fixed-router control starts bit-for-bit from
        # the exact same model, then only its address tables are frozen.
        models['phase_fixed_router'].load_state_dict(models['phase_graph'].state_dict())
        models = {name: model.to(device) for name, model in models.items()}
        initial_routes = models['phase_graph'].routes.weight.detach().argmax(-1).cpu()
        optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
                      for name, model in models.items()}
        streams = {scenario: torch.Generator().manual_seed(seed * 10001 + j*989)
                   for scenario, j in TRAIN_IDS.items()}
        losses = []
        wall = time.perf_counter()
        for step in range(args.steps):
            scenario = 'grouped' if step % 2 else 'independent'
            seq_length = args.length
            batch = episodes(args.batch, seq_length, streams[scenario], device=device, scenario=scenario)
            step_record = {'step': step+1, 'scenario': scenario}
            for name, model in models.items():
                model.train()
                optimizers[name].zero_grad(set_to_none=True)
                output, state = model_logits(model, batch)
                loss = loss_for(output, batch.target, name)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizers[name].step()
                step_record[name] = round(loss.detach().item(), 6)
            if (step + 1) % max(1, args.steps // 10) == 0:
                losses.append(step_record)
        train_seconds = round(time.perf_counter()-wall, 3)
        rows = evaluate(models, device, seed=seed * 100000 + 900000,
                        cases=args.eval_cases, lengths=args.eval_lengths)
        for row in rows:
            row['seed'] = seed
        diag = models['phase_graph'].route_diagnostics()
        current_routes = models['phase_graph'].routes.weight.detach().argmax(-1).cpu()
        diag['writer_top_sites_changed_during_training'] = int((initial_routes != current_routes).sum().item())
        diag['key_count'] = int(initial_routes.numel())
        # Exact algebraic equivalence of a hard-route paired ratio and a
        # directly addressed phase. Quantifies the *absence* of extra bits.
        with torch.no_grad():
            proof = episodes(32, args.length, torch.Generator().manual_seed(seed+1234567),device=device)
            g = models['phase_graph']
            m = g.write_state(proof.keys,proof.bits)
            za,zb = g.waves(m,proof.query)
            a = g.address(proof.query,read=True)
            source_phase = (m[:,1:] * a).sum(-1)*(2*math.pi)
            analytic = torch.complex(torch.cos(source_phase),torch.sin(source_phase))
            diag['ratio_minus_direct_max_complex_abs'] = float(((za*zb.conj())-analytic).abs().max().item())
        # State quantization and actual stored-byte count for all memory types.
        one = episodes(3, args.length, torch.Generator().manual_seed(seed+880000),device=device)
        with torch.no_grad():
            observed = {}
            for name, model in models.items():
                _, state = model_logits(model, one)
                if isinstance(state, tuple):
                    observed[name]={'state_bytes':64,'shared_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad), 'shared_total_parameters':sum(p.numel() for p in model.parameters())}
                    continue
                if name in ('phase_graph','phase_fixed_router'):
                    b = (state * 255).round().to(torch.uint8)
                else:
                    b = ((state.clamp(-1,1)+1)*127.5).round().to(torch.uint8)
                observed[name] = {'state_bytes': int(b[0].numel()),
                                  'shared_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad), 'shared_total_parameters':sum(p.numel() for p in model.parameters()),
                                  'sample_byte_checksum':int(b[0].to(torch.int).sum().item())}
        receipt = {'seed':seed, 'device':device, 'steps':args.steps,'batch_size':args.batch,
                   'sequence_length_train':args.length, 'lr':args.lr,'train_seconds':train_seconds,
                   'model_properties':observed,'route_diagnostics':diag,
                   'train_loss_samples':losses,'test':rows,
                   'caveats':[
                     'Learned router over a fixed known 128-key universe; not OOD key generalization.',
                     'Coherent soft mixtures over 63 paths; counts as many path contributions, not one physical launch.',
                     'Two complex end-port features, potentially greater read compute than a direct 64-byte address.',
                     'Synthetic binary fact retrieval, NOT language modelling or Transformer replacement.',
                     '64-byte phase, GRU, fast weights and token tape; shared checkpoint parameters reported separately.',
                     'Exact 16-byte indexed oracle exists because key domain is public and known.',
                     'Reader comparisons share the trained phase writer, but this is not matched-FLOP training.',
                     'Training alternates independent and grouped streams; broken-group is evaluation-only.',
                     'Grouped labels remain fixed within an episode; no unobserved group inference is attempted.',
                   ]}
        (outdir/f'seed{seed}.json').write_text(json.dumps(receipt, indent=2)+'\n')
        seed_receipts.append({'seed': seed, 'seconds':train_seconds,
                              'independent_64':{f'{r["model"]}:{r["readout"]}':r['accuracy'] for r in rows if r['scenario']=='independent' and r['length']==64}})
        if args.checkpoints:
            torch.save({'seed':seed,'step':args.steps, 'models': {n:m.state_dict() for n,m in models.items()},
                        'optimizers':{n:o.state_dict() for n,o in optimizers.items()}}, outdir/f'seed{seed}.pt')
        print(json.dumps(seed_receipts[-1]), flush=True)
    (outdir/'summary.json').write_text(json.dumps({'gate':'Gate 1 exploratory','runs':seed_receipts},indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch', type=int, default=96)
    parser.add_argument('--length', type=int, default=24)
    parser.add_argument('--eval-cases', type=int, default=512)
    parser.add_argument('--eval-lengths', type=int, nargs='+', default=[16,32,64,96])
    parser.add_argument('--lr',type=float, default=.002)
    parser.add_argument('--seeds',type=int,nargs='+',default=[41,97,127])
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    parser.add_argument('--out', default='results/gate1')
    parser.add_argument('--checkpoints',action='store_true')
    args = parser.parse_args()
    if args.smoke:
        args.steps, args.batch, args.length, args.eval_cases, args.eval_lengths, args.seeds = 2, 4, 8, 12, [8,16], [41]
    main(args)
