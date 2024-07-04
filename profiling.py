import jax
import flax.linen.module as module_lib
from flax.linen.module import _CallInfo
from flax.linen.summary import (
    _process_inputs,
    _get_module_variables,
    _get_call_flops,
)
from typing import Set, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ModuleCall:
    path: Tuple[str, ...]
    method: str
    flops: float
    vjp_flops: float


def trace_module_calls(module: module_lib.Module, *args, **kwargs) -> List[ModuleCall]:
    """
    Get the FLOPs estimate and parameter count for a Flax module.
    """
    with module_lib._tabulate_context():

        def _get_variables():
            return module.init(*args, **kwargs)

        variables = jax.eval_shape(_get_variables)
        calls = module_lib._context.call_info_stack[-1].calls
        calls.sort(key=lambda c: c.index)

    all_paths: Set[Tuple[str, ...]] = set(call.path for call in calls)
    visited_paths: Set[Tuple[str, ...]] = set()
    calls_out: List[ModuleCall] = []

    for c in calls:
        inputs = _process_inputs(c.args, c.kwargs)

        visited_paths.add(c.path)
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
