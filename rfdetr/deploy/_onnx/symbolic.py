# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""
CustomOpSymbolicRegistry class
"""

from typing import List, Callable, Any


class CustomOpSymbolicRegistry:
    # _SYMBOLICS = {}
    _OPTIMIZER: List[Callable[..., Any]] = []

    @classmethod
    def optimizer(cls, fn: Callable[..., Any]) -> None:
        cls._OPTIMIZER.append(fn)


def register_optimizer() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def optimizer_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
        CustomOpSymbolicRegistry.optimizer(fn)
        return fn

    return optimizer_wrapper
