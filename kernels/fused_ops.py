import torch
import triton
import triton.language as tl

@triton.jit
def _fused_add_rms_norm_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr,
    stride_x, stride_res, stride_out,
    N, eps, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + row_idx * stride_x + cols, mask=mask, other=0.0)
    res = tl.load(residual_ptr + row_idx * stride_res + cols, mask=mask, other=0.0)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)

    # fused add
    x_new = x + res
    tl.store(residual_ptr + row_idx * stride_res + cols, x_new, mask=mask)

    # rms norm
    variance = tl.sum(x_new * x_new, axis=0) / N
    rsqrt = 1.0 / tl.sqrt(variance + eps)
    out = x_new * rsqrt * w

    tl.store(out_ptr + row_idx * stride_out + cols, out, mask=mask)


def triton_fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    _fused_add_rms_norm_kernel[(M,)](
        x, residual, weight, out,
        x.stride(0), residual.stride(0), out.stride(0),
        N, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return out, residual


class TritonEngine:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.is_ready = torch.cuda.is_available()

    def apply_fused_norm(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
        if not self.enabled or not self.is_ready:
            # fallback torch
            res = x + residual
            variance = res.pow(2).mean(-1, keepdim=True)
            out = res * torch.rsqrt(variance + 1e-6) * weight
            return out, res

        return triton_fused_add_rms_norm(x, residual, weight)