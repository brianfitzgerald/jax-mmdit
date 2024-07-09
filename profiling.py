import jax
import flax.linen.module as module_lib
from flax.linen.summary import (
    _get_call_flops,
    _bytes_repr
)
from typing import Tuple, List
from dataclasses import dataclass
from flax.linen import Conv


@dataclass
class ModuleCall:
    path: Tuple[str, ...]
    method: str
    flops: float
    vjp_flops: float


DEVICE_PEAK_FLOPS = {
    "NVIDIA H100 80GB HBM3": {
        "fp32": 5.1e13,
        "fp16": 1.513e15,
    }
}


def get_peak_flops() -> float:
    device_kind = jax.devices()[0].device_kind
    peak_flops = DEVICE_PEAK_FLOPS[device_kind]["fp32"]
    return peak_flops


def trace_module_calls(module: module_lib.Module, *args, **kwargs) -> List[ModuleCall]:
    """
    Get the FLOPs estimate and parameter count for a Flax module.
    """

    with module_lib._tabulate_context():

        def _get_variables():
            return module.init(*args, **kwargs)

        jax.eval_shape(_get_variables)
        calls = module_lib._context.call_info_stack[-1].calls
        calls.sort(key=lambda c: c.index)

    calls_out: List[ModuleCall] = []

    for c in calls:
        flops, vjp_flops = _get_call_flops(c, True, True)
        calls_out.append(
            ModuleCall(
                c.path,
                c.method,
                flops,
                vjp_flops,
            )
        )

    return calls_out


def memory_usage_params(model_params):
    total_bytes, total_params = 0, 0
    for param in jax.tree_leaves(model_params):
        total_bytes += param.size * param.dtype.itemsize
        total_params += param.size
    total_bytes = _bytes_repr(total_bytes)
    return total_bytes, total_params
