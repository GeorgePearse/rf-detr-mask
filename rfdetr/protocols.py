# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class Exportable(Protocol):
    """Protocol for objects that can be exported to PyTorch format."""

    export_torch: bool
    export_dir: Optional[Path]

    def export(self, *args: Any, **kwargs: Any) -> None:
        """Export the model to file."""
        ...


@runtime_checkable
class HasExport(Protocol):
    """Protocol for objects that have an export method."""

    def export(self, *args: Any, **kwargs: Any) -> None:
        """Export the model to file."""
        ...


@runtime_checkable
class HasModelDump(Protocol):
    """Protocol for objects that have a model_dump method (like Pydantic models)."""

    def model_dump(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        ...
