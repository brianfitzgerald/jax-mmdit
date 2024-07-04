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
    index: int
    path: Tuple[str, ...]
    mutable: bool
    method: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    outputs: Any
    flops: float
    vjp_flops: float


def trace_module_calls(module: module_lib.Module, *args, **kwargs) -> float:
    """
    Get the FLOPs estimate and parameter count for a Flax module.
    """
    with module_lib._tabulate_context():

        def _get_variables():
            return module.init(*args, **kwargs)

        variables = jax.eval_shape(_get_variables)
        calls = module_lib._context.call_info_stack[-1].calls
        calls.sort(key=lambda c: c.index)

    collections: Set[str] = set(variables.keys())
    calls: List[_CallInfo] = []
    all_paths: Set[Tuple[str, ...]] = set(call.path for call in calls)
    visited_paths: Set[Tuple[str, ...]] = set()

    for c in calls:
        inputs = _process_inputs(c.args, c.kwargs)

        if c.path in visited_paths or not hasattr(c.module, c.method):
            module_vars = {}
            counted_vars = {}
        else:
            module_vars, _ = _get_module_variables(c.path, variables, all_paths)
            counted_vars = module_vars

        visited_paths.add(c.path)
        calls.append(
            ModuleCall(
                c.path,
                c.method,
                inputs,
                c.outputs,
                module_vars,
                counted_vars,
                *_get_call_flops(c, True, True),
            )
        )

    return calls
