"""FLOP accounting in the multiply-accumulate (MAC) convention."""

import contextlib

_STATE = {"active": False, "macs": 0}


def record(macs):
    """Adds multiply-accumulates of sparse or gather-based operations that dense counters miss."""
    if _STATE["active"]:
        _STATE["macs"] += int(macs)


@contextlib.contextmanager
def count_macs():
    """Counts dense matmul/convolution MACs plus explicitly recorded sparse MACs.

    Usage::

        with count_macs() as result:
            model(inputs)
        print(result["total"])
    """
    from torch.utils.flop_counter import FlopCounterMode

    _STATE["active"] = True
    _STATE["macs"] = 0
    result = {}
    counter = FlopCounterMode(display=False)
    try:
        with counter:
            yield result
    finally:
        _STATE["active"] = False
        result["dense"] = int(counter.get_total_flops()) // 2
        result["sparse"] = int(_STATE["macs"])
        result["total"] = result["dense"] + result["sparse"]
