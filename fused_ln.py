"""FusedLayerNorm — custom CUDA LayerNorm with a real backward pass.

This module replaces the ad-hoc kernel dispatch that used to live inside
`NanoLLM.py`. Three things changed, and each fixes a concrete defect:

1. **bf16/fp16 support.** The old dispatch only fired for float32, so under
   `torch.autocast(dtype=torch.bfloat16)` the custom kernels were never
   actually used during training — every LayerNorm silently fell back to
   `F.layer_norm`. The kernels in `fused_layernorm_train.cu` are templated on
   the scalar type, so they now run on the real training hot path.

2. **Backward pass.** The benchmark kernels are forward-only. Autograd could
   not have flowed through them at all. `FusedLayerNormFn` below binds a
   hand-written dx/dgamma/dbeta kernel.

3. **No silent failures.** Extension compilation used to be wrapped in a bare
   `except: pass`, so a missing CUDA toolchain degraded to PyTorch LayerNorm
   with no indication. Now it warns once, loudly, and `kernel_status()` reports
   what is actually in use.

The kernel keeps gamma/beta (and all accumulators) in float32 even when the
activations are bf16 — reducing a row in bf16 loses far too much precision.
"""

import os
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_ext = None
_load_error = None  # type: Optional[Exception]
_load_attempted = False

#: dtypes the custom kernels handle natively.
_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def load_extension(verbose: bool = False):
    """JIT-compile (or fetch from cache) the fused LayerNorm extension.

    Compilation is deferred to first use rather than import time so that
    importing the model on a CPU-only box stays instant and side-effect free.
    """
    global _ext, _load_error, _load_attempted
    if _load_attempted:
        return _ext
    _load_attempted = True

    if not torch.cuda.is_available():
        _load_error = RuntimeError("CUDA not available")
        return None

    try:
        from torch.utils.cpp_extension import load

        _ext = load(
            name="fused_ln_train",
            sources=[os.path.join(_THIS_DIR, "fused_layernorm_train.cu")],
            # Deliberately no --use_fast_math: it turns the Welford `delta /
            # cnt` into an approximate reciprocal, which shows up as real error
            # against PyTorch's LayerNorm for a marginal speed gain.
            extra_cuda_cflags=["-O3"],
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001 - we deliberately degrade gracefully
        _load_error = exc
        warnings.warn(
            f"FusedLayerNorm: CUDA extension failed to build ({exc.__class__.__name__}: {exc}). "
            "Falling back to F.layer_norm. Install `ninja` and a matching CUDA "
            "toolkit to use the custom kernels.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _ext


def kernel_status() -> str:
    """Human-readable description of which LayerNorm path is active."""
    if _ext is not None:
        return "fused CUDA kernel (fp32/bf16/fp16, custom backward)"
    if _load_error is not None:
        return f"F.layer_norm fallback ({_load_error.__class__.__name__}: {_load_error})"
    return "F.layer_norm fallback (extension not loaded yet)"


# ─────────────────────────────────────────────────────────────────────────────
# torch.library registration
#
# Calling a pybind11 extension directly from inside a torch.compile region
# forces a graph break at every LayerNorm. Registering the kernels as custom
# ops with fake (meta) implementations lets Dynamo trace straight through them,
# so the compiled graph stays whole.
# ─────────────────────────────────────────────────────────────────────────────

@torch.library.custom_op("nanollm::fused_ln_fwd", mutates_args=())
def _fused_ln_fwd(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out, mean, rstd = _ext.forward(x, gamma, beta, eps)
    return out, mean, rstd


@_fused_ln_fwd.register_fake
def _(x, gamma, beta, eps):
    rows = x.shape[0]
    return (
        torch.empty_like(x),
        x.new_empty((rows,), dtype=torch.float32),
        x.new_empty((rows,), dtype=torch.float32),
    )


@torch.library.custom_op("nanollm::fused_ln_bwd", mutates_args=())
def _fused_ln_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx, dgamma, dbeta = _ext.backward(dy, x, gamma, mean, rstd)
    return dx, dgamma, dbeta


@_fused_ln_bwd.register_fake
def _(dy, x, gamma, mean, rstd):
    return (
        torch.empty_like(x),
        torch.empty_like(gamma),
        torch.empty_like(gamma),
    )


class FusedLayerNormFn(torch.autograd.Function):
    """Autograd binding for the fused kernels."""

    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        out, mean, rstd = torch.ops.nanollm.fused_ln_fwd(x, gamma, beta, eps)
        ctx.save_for_backward(x, gamma, mean, rstd)
        return out

    @staticmethod
    def backward(ctx, dy):
        x, gamma, mean, rstd = ctx.saved_tensors
        dx, dgamma, dbeta = torch.ops.nanollm.fused_ln_bwd(
            dy.contiguous(), x, gamma, mean, rstd
        )
        return dx, dgamma, dbeta, None


class FusedLayerNorm(nn.Module):
    """Drop-in replacement for :class:`torch.nn.LayerNorm` over the last dim.

    Args:
        normalized_shape: size of the trailing dimension to normalise.
        eps: numerical stabiliser added to the variance.
        force_fallback: skip the custom kernel entirely (used by benchmarks and
            for A/B-ing kernel vs. PyTorch during a real training run).
    """

    def __init__(self, normalized_shape, eps: float = 1e-5, force_fallback: bool = False):
        super().__init__()
        if isinstance(normalized_shape, (tuple, list)):
            if len(normalized_shape) != 1:
                raise ValueError("FusedLayerNorm only normalises a single trailing dim")
            normalized_shape = normalized_shape[0]
        self.normalized_shape = int(normalized_shape)
        self.eps = eps
        self.force_fallback = force_fallback
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def _use_kernel(self, x: torch.Tensor) -> bool:
        if self.force_fallback or not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES:
            return False
        return load_extension() is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_kernel(x):
            return F.layer_norm(
                x,
                (self.normalized_shape,),
                self.weight.to(x.dtype),
                self.bias.to(x.dtype),
                self.eps,
            )

        # The kernel requires float32 affine params regardless of activation
        # dtype; under autocast these are already float32 so this is a no-op.
        gamma = self.weight if self.weight.dtype == torch.float32 else self.weight.float()
        beta = self.bias if self.bias.dtype == torch.float32 else self.bias.float()

        shape = x.shape
        x_2d = x.reshape(-1, self.normalized_shape)
        out = FusedLayerNormFn.apply(x_2d, gamma, beta, self.eps)
        return out.view(shape)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"
