# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# Import standard library
import os

# Import model classes
from rfdetr.detr import RFDETRBase, RFDETRLarge
from rfdetr.model_config import ModelConfig

# Configure MPS fallback if needed
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Initialize logging system
from rfdetr.util.init_logs import init_logs_directory
from rfdetr.util.logging_config import setup_logging

# Setup logs directory and configure logging
init_logs_directory()
setup_logging()
