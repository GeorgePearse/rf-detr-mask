"""Training module with refactored training logic."""

from rfdetr.core.training.trainer import Trainer
from rfdetr.core.training.callbacks import CallbackManager

__all__ = ["Trainer", "CallbackManager"]