from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class TruLensTracer:
    def trace(self, name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return fn(*args, **kwargs)
