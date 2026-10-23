"""
Torch-TensorRT / TensorRT: sibling FP8 GEMMs with identical input scales might give an all-NaN engine.

Two linears (gate/up of a Qwen3-style MLP) read the same activation, so ModelOpt calibrates the
same input amax for both, while their weight amax differ. Compiled with the Q/DQ nodes ModelOpt
exports, the engine output is NaN. Any one of these makes it correct again:
  --fix input   : 0.1% different input amax for the up projection
  --fix weight  : the same weight amax for both projections

usage: python fp8_sibling_nan.py [--fix none|input|weight]
needs: torch, nvidia-modelopt, torch-tensorrt (FP8 capable GPU, e.g. sm89+)
"""

import enum

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt
import typer
from modelopt.torch.quantization.utils import export_torch_mode

H, FF, L = 4096, 12288, 64


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, FF, bias=False)
        self.up = nn.Linear(H, FF, bias=False)
        self.down = nn.Linear(FF, H, bias=False)

    def forward(self, x):
        return x + self.down(F.silu(self.gate(x)) * self.up(x))


class Fix(enum.StrEnum):
    none = "none"
    input = "input"
    weight = "weight"


def main(
    fix: Fix = typer.Option(Fix.none, help="workaround to apply before compiling"),
) -> None:
    torch.manual_seed(0)
    m = MLP().cuda().half().eval()
    with torch.no_grad():
        m.up.weight.mul_(0.6)  # different weight amax than gate
    x = torch.randn(1, L, H, device="cuda", dtype=torch.half)
    mtq.quantize(m, mtq.FP8_DEFAULT_CFG, forward_loop=lambda mod: mod(x))

    if fix == Fix.input:
        m.up.input_quantizer.amax = m.up.input_quantizer.amax * 1.001
    elif fix == Fix.weight:
        a = torch.maximum(m.gate.weight_quantizer.amax, m.up.weight_quantizer.amax)
        m.gate.weight_quantizer.amax = a.clone()
        m.up.weight_quantizer.amax = a.clone()

    print(
        "input amax gate/up:",
        m.gate.input_quantizer.amax.item(),
        m.up.input_quantizer.amax.item(),
        "| weight amax gate/up:",
        m.gate.weight_quantizer.amax.item(),
        m.up.weight_quantizer.amax.item(),
    )
    with torch.no_grad():
        ref = m(x)
        with export_torch_mode():
            ep = torch.export.export(m, (x,), strict=False)
        trt = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[x],
            enabled_precisions={torch.half},
            use_explicit_typing=False,
            min_block_size=1,
        )
        out = trt(x)
    cos = F.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    print(
        f"fix={fix.value}: engine output finite={torch.isfinite(out).all().item()} cosine vs eager={cos:.5f}"
    )


if __name__ == "__main__":
    typer.run(main)
