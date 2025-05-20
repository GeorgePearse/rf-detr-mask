# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

import os
from typing import Optional

import roboflow
import torch
from rf100vl import get_rf100vl_projects

from rfdetr import RFDETRBase
from rfdetr.config_utils import load_config


def download_dataset(rf_project: roboflow.Project, dataset_version: int):
    versions = rf_project.versions()
    if dataset_version is not None:
        versions = [v for v in versions if v.version == str(dataset_version)]
        if len(versions) == 0:
            raise ValueError(f"Dataset version {dataset_version} not found")
        version = versions[0]
    else:
        version = max(versions, key=lambda v: v.id)
    location = os.path.join("datasets/", rf_project.name + "_v" + version.version)
    if not os.path.exists(location):
        location = version.download(
            model_format="coco", location=location, overwrite=False
        ).location

    return location


def train_from_rf_project(rf_project: roboflow.Project, dataset_version: int):
    location = download_dataset(rf_project, dataset_version)
    print(location)
    rf_detr = RFDETRBase()
    device_supports_cuda = torch.cuda.is_available()
    rf_detr.train(
        dataset_dir=location,
        epochs=1,
        device="cuda" if device_supports_cuda else "cpu",
    )


def train_from_coco_dir(coco_dir: str):
    rf_detr = RFDETRBase()
    rf_detr.train(
        dataset_dir=coco_dir,
        epochs=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def trainer(coco_dir: Optional[str] = None, 
         api_key: Optional[str] = None,
         workspace: Optional[str] = None,
         project_name: Optional[str] = None,
         dataset_version: Optional[int] = None,
         config_path: str = "configs/default.yaml"):
    
    # Load configuration from YAML file
    config = load_config(config_path)
    
    if coco_dir is not None:
        train_from_coco_dir(coco_dir)
        return

    if (workspace is None and project_name is not None) or (
        workspace is not None and project_name is None
    ):
        raise ValueError("Either both workspace and project_name must be provided or none of them")

    if workspace is not None:
        rf = roboflow.Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
    else:
        projects = get_rf100vl_projects(api_key=api_key)
        project = projects[0].rf_project

    train_from_rf_project(project, dataset_version)


if __name__ == "__main__":
    import sys
    
    # Simple argument parsing for backward compatibility
    coco_dir = None
    api_key = None
    workspace = None
    project_name = None
    dataset_version = None
    config_path = "configs/default.yaml"
    
    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--coco_dir" and i+1 < len(sys.argv):
            coco_dir = sys.argv[i+1]
            i += 2
        elif arg == "--api_key" and i+1 < len(sys.argv):
            api_key = sys.argv[i+1]
            i += 2
        elif arg == "--workspace" and i+1 < len(sys.argv):
            workspace = sys.argv[i+1]
            i += 2
        elif arg == "--project_name" and i+1 < len(sys.argv):
            project_name = sys.argv[i+1]
            i += 2
        elif arg == "--dataset_version" and i+1 < len(sys.argv):
            dataset_version = int(sys.argv[i+1])
            i += 2
        elif arg == "--config" and i+1 < len(sys.argv):
            config_path = sys.argv[i+1]
            i += 2
        else:
            i += 1
    
    trainer(
        coco_dir=coco_dir,
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        dataset_version=dataset_version,
        config_path=config_path
    )
