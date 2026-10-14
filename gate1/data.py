"""Synthetic online-memory episodes, generated afresh with disjoint RNG streams."""
from __future__ import annotations
import torch
from dataclasses import dataclass

NKEYS = 128


@dataclass
class Batch:
    keys: torch.Tensor
    bits: torch.Tensor
    query: torch.Tensor
    target: torch.Tensor


def episodes(batch: int, length: int, generator: torch.Generator, *,
             device: str = 'cpu', scenario: str = 'independent', nkeys: int = NKEYS) -> Batch:
    """Question is a previously written key; target is its last written bit.

    'grouped' reuses one latent Bernoulli bit per group of eight keys.
    'broken_group' uses independent bits (evaluated separately under a
    grouped-trained model). 'rewrite' forces the last write to overwrite
    an earlier key with its opposite bit and queries that key.
    """
    if scenario not in ('independent', 'grouped', 'broken_group', 'rewrite', 'grouped16'):
        raise ValueError(scenario)
    if batch <= 0 or length < 2:
        raise ValueError('batch and length must be positive; length >=2')
    keys = torch.randint(nkeys, (batch, length), generator=generator)
    if scenario.startswith('grouped'):
        size = 16 if scenario == 'grouped16' else 8
        group_values = torch.randint(0, 2, (batch, (nkeys+size-1)//size), generator=generator)
        bits = group_values.gather(1, keys // size)
    else:
        bits = torch.randint(0, 2, (batch, length), generator=generator)
    batch_rows = torch.arange(batch)
    if scenario == 'rewrite':
        earlier = torch.randint(0, length-1, (batch,), generator=generator)
        keys[batch_rows, -1] = keys[batch_rows, earlier]
        # Opposite of the most recent PREVIOUS write for that key.
        previous = (keys[:, :-1] == keys[:, -1, None])
        last_idx = torch.where(previous, torch.arange(length-1), -1).max(1).values
        last_val = bits[batch_rows, last_idx]
        bits[:, -1] = 1 - last_val
        query = keys[:, -1].clone()
    else:
        positions = torch.randint(length, (batch,), generator=generator)
        query = keys[batch_rows, positions].clone()
    match = keys.eq(query[:, None])
    last_idx = torch.where(match, torch.arange(length), -1).max(1).values
    assert bool((last_idx >= 0).all())
    target = bits[batch_rows, last_idx].clone()
    return Batch(keys.to(device), bits.to(device), query.to(device), target.to(device))
