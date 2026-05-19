from __future__ import annotations

from inverse.checkpoint import LoadedScoreCheckpoint
from inverse.methods.base import SamplerParams
from inverse.operators import LinearOperator


def sample(
    checkpoint: LoadedScoreCheckpoint,
    operator: LinearOperator,
    y_norm,
    params: SamplerParams,
):
    raise NotImplementedError("DDNM sampler belongs in inverse/methods/ddnm.py.")
