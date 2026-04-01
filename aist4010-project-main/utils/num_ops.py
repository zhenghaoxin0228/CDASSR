# -*- coding: utf-8 -*-
# File: utils/num_ops.py


def clamp(num: float, a: float, b: float) -> float:
    """
    Restrict a number to a specific range.

    Args:
        num (Real): Target number.
        a (Real): Lower bound of the range.
        b (Real): Upper bound of the range.

    Returns:
        Real: `num` clamped between `a` and `b`.
    """

    assert a <= b, f"{a} <= {b}. `a` must not be greater than `b`."

    return max(a, min(num, b))
