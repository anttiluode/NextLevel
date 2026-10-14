"""Gate 1 trainable memory systems. All recurrent states quantized to uint8.

The graph model is a *computational phase-path toy*, not an electromagnetic
wave simulation. The learnt route is a distribution over simultaneous launch
sites, so its physical equivalent would require many launches. We report the
64-state-byte cost but do not call it one-read, one-port hardware.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

SLOTS = 64
PORT = 0


def q8(x: torch.Tensor, signed: bool = False) -> torch.Tensor:
    """Straight-through 8-bit quantizer; evaluation stores its exact uint8 code."""
    if signed:
        v = ((x.clamp(-1, 1) + 1) * 0.5 * 255)
        quant = v.round() / 255 * 2 - 1
    else:
        quant = (x.clamp(0, 1) * 255).round() / 255
    return x + (quant - x).detach()


def build_paths() -> tuple[torch.Tensor, torch.Tensor]:
    """12-step walks to fixed port, identical to Gate 0's toroidal routing."""
    from pathlib import Path
    import sys
    root = str(Path(__file__).resolve().parents[1] / 'src')
    if root not in sys.path:
        sys.path.insert(0, root)
    from nextlevel.experiment import route_to_port
    before, after = [], []
    for site in range(1, SLOTS):
        path = route_to_port(site)
        a = torch.bincount(torch.tensor(path[:-1]), minlength=SLOTS).float()
        b = torch.bincount(torch.tensor(path[1:]), minlength=SLOTS).float()
        before.append(a)
        after.append(b)
    return torch.stack(before), torch.stack(after)


class PhaseGraph(nn.Module):
    """Write experiences into 64 byte-valued phases, interrogate by learned paths.

    The router learns a probability for each of 63 launch sites for every key;
    the writer uses the same learned geometry. This is continuous multi-path
    interrogation; routing cost is 63 simultaneous hypothetical paths, NOT
    a single adaptive hardware ray. There are two end-port complex readings
    after coherent mixing of paths, with a paired/ratio/raw/fixed readout.
    """
    state_bytes = 64

    def __init__(self, nkeys: int = 128, hidden: int = 48, temperature: float = 0.5, freeze_routes: bool = False):
        super().__init__()
        self.nkeys = nkeys
        self.temperature = temperature
        self.routes = nn.Embedding(nkeys, SLOTS - 1)
        nn.init.normal_(self.routes.weight, 0.0, 0.05)
        # A separately trainable interrogation policy: a query-specific
        # correction to the writer address and a preference for probe order.
        self.read_delta = nn.Embedding(nkeys, SLOTS - 1)
        nn.init.zeros_(self.read_delta.weight)
        self.order_logits = nn.Embedding(nkeys, 2)
        nn.init.zeros_(self.order_logits.weight)
        if freeze_routes:
            self.routes.weight.requires_grad_(False)
            self.read_delta.weight.requires_grad_(False)
        self.bit_phase = nn.Parameter(torch.tensor([-1.1, 1.1]))
        self.ratio_head = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.paired_head = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.single_head = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.fixed_head = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.policy_head = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.direct_head = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        before, after = build_paths()
        self.register_buffer('before', before)
        self.register_buffer('after', after)

    def address(self, keys: torch.Tensor, *, read: bool = False) -> torch.Tensor:
        logits = self.routes(keys)
        if read:
            logits = logits + self.read_delta(keys)
        soft = F.softmax(logits / self.temperature, dim=-1)
        # Train hard addresses with soft derivatives; inference has one actual
        # launch site per query and all writes do a one-cell update.
        hard = F.one_hot(soft.argmax(-1), num_classes=SLOTS-1).to(soft.dtype)
        return hard + soft - soft.detach()

    def write_state(self, keys: torch.Tensor, bits: torch.Tensor, *, quantize: bool = True) -> torch.Tensor:
        batch, length = keys.shape
        state = torch.zeros(batch, SLOTS, device=keys.device)
        for t in range(length):
            address = self.address(keys[:, t])  # [B,63]
            # One shared learned bit->phase code; the site code is the *state*.
            value = torch.sigmoid(self.bit_phase[bits[:, t]])
            slots = state[:, 1:]
            # Soft writing to all sites: learnable, nontrivial sharing/interference.
            new = slots * (1 - address) + address * value[:, None]
            state = torch.cat([state[:, :1] * 0, new], dim=-1)
            if quantize:
                state = q8(state)
        return state

    def waves(self, state: torch.Tensor, query: torch.Tensor, *, fixed_route: bool = False):
        phase = state * (2.0 * math.pi)
        before = phase @ self.before.T
        after = phase @ self.after.T
        wave_a = torch.complex(torch.cos(before), torch.sin(before))
        wave_b = torch.complex(torch.cos(after), torch.sin(after))
        address = torch.full_like(wave_a.real, 1. / (SLOTS - 1)) if fixed_route else self.address(query, read=True)
        z_a = (wave_a * address).sum(-1)
        z_b = (wave_b * address).sum(-1)
        return z_a, z_b

    @staticmethod
    def features(za, zb, kind: str) -> torch.Tensor:
        if kind == 'ratio':
            # Coherent path mixtures are *not* guaranteed to cancel a shared
            # phase exactly; compare raw pair rather than asserting invariance.
            r = za * zb.conj()
            return torch.stack([r.real, r.imag, torch.abs(za), torch.abs(zb)], dim=-1)
        if kind == 'paired':
            return torch.stack([za.real, za.imag, zb.real, zb.imag], dim=-1)
        if kind == 'single':
            return torch.stack([za.real, za.imag, torch.abs(za), torch.zeros_like(za.real)], dim=-1)
        raise KeyError(kind)

    def forward(self, keys: torch.Tensor, bits: torch.Tensor, query: torch.Tensor):
        state = self.write_state(keys, bits)
        za, zb = self.waves(state, query)
        fixed_a, fixed_b = self.waves(state, query, fixed_route=True)
        order_p = F.softmax(self.order_logits(query), dim=-1)
        policy_wave = order_p[:, 0] * za + order_p[:, 1] * zb
        # Strong control: address the source phase directly instead of
        # performing 24 path-steps to infer it by an order-pair ratio.
        address = self.address(query, read=True)
        source = (state[:,1:] * address).sum(-1) * (2 * math.pi)
        direct = torch.stack([torch.cos(source),torch.sin(source),
                              torch.ones_like(source), torch.zeros_like(source)],dim=-1)
        return {
            'ratio': self.ratio_head(self.features(za, zb, 'ratio')).squeeze(-1),
            'paired': self.paired_head(self.features(za, zb, 'paired')).squeeze(-1),
            'single': self.single_head(self.features(za, zb, 'single')).squeeze(-1),
            'fixed_route': self.fixed_head(self.features(fixed_a, fixed_b, 'paired')).squeeze(-1),
            'learned_order_policy': self.policy_head(self.features(policy_wave, policy_wave, 'paired')).squeeze(-1),
            'direct_addressed': self.direct_head(direct).squeeze(-1),
        }, state

    @torch.no_grad()
    def route_diagnostics(self) -> dict:
        p = F.softmax(self.routes.weight / self.temperature, dim=-1)
        bins = p.argmax(dim=-1)
        entropy = -(p * p.clamp_min(1e-9).log()).sum(-1).mean().item()
        r = F.softmax((self.routes.weight + self.read_delta.weight)/self.temperature,dim=-1)
        agreement = (bins == r.argmax(-1)).float().mean().item()
        orders = F.softmax(self.order_logits.weight, dim=-1)
        return {'writer_route_entropy_nats': entropy,
                'distinct_writer_top_sites': int(bins.unique().numel()),
                'mean_writer_site_probability': float(p.max(dim=-1).values.mean().item()),
                'read_write_top_site_agreement': agreement,
                'mean_order_ab_probability': float(orders[:,0].mean().item()),
                'mean_order_confidence':float(orders.max(-1).values.mean().item())}


class FixedRouterPhaseGraph(PhaseGraph):
    """Same phase-path memory with a random, permanently frozen key/site map."""
    def __init__(self, **kw):
        super().__init__(freeze_routes=True, **kw)


class GRUMemory(nn.Module):
    """64-byte quantized recurrent activation state (trained GRU)."""
    state_bytes = 64

    def __init__(self, nkeys: int = 128, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(nkeys, 24)
        self.gru = nn.GRUCell(25, hidden)
        self.read_embed = nn.Embedding(nkeys, 24)
        self.head = nn.Sequential(nn.Linear(hidden + 24, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, keys, bits, query):
        state = torch.zeros(keys.size(0), 64, device=keys.device)
        for t in range(keys.size(1)):
            x = torch.cat([self.embed(keys[:, t]), bits[:, t, None].float() * 2 - 1], dim=-1)
            state = q8(self.gru(x, state), signed=True)
        inp = torch.cat([state, self.read_embed(query)], dim=-1)
        return {'gru': self.head(inp).squeeze(-1)}, state


class FastWeightMemory(nn.Module):
    """64-byte rank-one update matrix (8x8); learned key and query maps."""
    state_bytes = 64

    def __init__(self, nkeys: int = 128):
        super().__init__()
        self.writer_keys = nn.Embedding(nkeys, 8)
        self.reader_keys = nn.Embedding(nkeys, 8)
        self.value = nn.Embedding(2, 8)
        self.gate = nn.Parameter(torch.tensor(-2.))
        self.head = nn.Sequential(nn.Linear(8, 40), nn.Tanh(), nn.Linear(40, 1))

    def forward(self, keys, bits, query):
        state = torch.zeros(keys.size(0), 8, 8, device=keys.device)
        gate = torch.sigmoid(self.gate)
        for t in range(keys.size(1)):
            k = torch.tanh(self.writer_keys(keys[:, t]))
            v = torch.tanh(self.value(bits[:, t]))
            update = k.unsqueeze(-1) * v.unsqueeze(-2)
            state = q8((1 - gate) * state + gate * update, signed=True)
        r = torch.tanh(self.reader_keys(query))
        read = torch.bmm(r.unsqueeze(1), state).squeeze(1)
        return {'fast_weight': self.head(read).squeeze(-1)}, state


class LearnedWindowAttention(nn.Module):
    """Learned associative attention over the last 32 byte-pair records.

    Storage: 32 key bytes and 32 bit bytes. This is an *attention-based KV
    reader*, NOT a trained Transformer layer, and it has a finite recency window.
    """
    state_bytes = 64
    records = 32

    def __init__(self, nkeys: int = 128):
        super().__init__()
        self.embed = nn.Embedding(nkeys, 24)
        self.log_scale = nn.Parameter(torch.tensor(1.))
        self.recency = nn.Parameter(torch.tensor(0.))
        self.head = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, keys, bits, query):
        k, b = keys[:, -self.records:], bits[:, -self.records:]
        q = F.normalize(self.embed(query), dim=-1)
        stored = F.normalize(self.embed(k), dim=-1)
        logits = torch.einsum('bd,bld->bl', q, stored) * self.log_scale.exp().clamp(max=100)
        # Recency preference for the last matching write, trained rather than oracle.
        age = torch.arange(k.size(1), device=k.device).float()
        logits = logits + self.recency * (age / max(k.size(1)-1, 1))
        weights = logits.softmax(-1)
        bit = (weights * b.float()).sum(-1)
        peak = weights.max(-1).values
        return {'attention': self.head(torch.stack([bit, peak], dim=-1)).squeeze(-1)}, (k, b)


def exact_window(keys: torch.Tensor, bits: torch.Tensor, query: torch.Tensor, records: int = 32):
    """Strong deterministic latest matching write in the SAME 64-byte tape."""
    keys, bits = keys[:, -records:], bits[:, -records:]
    match = keys.eq(query[:, None])
    last = torch.where(match, torch.arange(keys.size(1), device=keys.device), -1).max(-1).values
    found = last >= 0
    chosen = bits.gather(1, last.clamp_min(0)[:, None]).squeeze(-1)
    return found, chosen


def exact_indexed(keys: torch.Tensor, bits: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """16-byte exact-known-domain oracle; no need to train and no key tags."""
    known = torch.zeros(keys.size(0), 128, dtype=torch.long, device=keys.device)
    for t in range(keys.size(1)):
        known.scatter_(1, keys[:, t, None], bits[:, t, None])
    return known.gather(1, query[:, None]).squeeze(1)
