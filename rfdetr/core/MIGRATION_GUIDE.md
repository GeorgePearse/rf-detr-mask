# RF-DETR Architecture Migration Guide

This guide explains how to migrate from the old god object pattern to the new refactored architecture.

## Overview of Changes

### Old Architecture (God Object)
- `RFDETR` class handled everything: configuration, model creation, training, inference, callbacks, metrics
- Tight coupling between components
- Difficult to test and extend
- Mixed configuration systems (argparse, Pydantic, dictionaries)

### New Architecture (SOLID Principles)
- **ConfigurationManager**: Unified configuration management
- **ModelFactory + ModelRegistry**: Model creation with dependency injection
- **CheckpointManager**: Fault-tolerant checkpoint management
- **Trainer**: Clean training orchestration
- **CallbackManager**: Extensible callback system
- Clear separation of concerns
- Easy to test, extend, and maintain

## Migration Steps

### 1. Configuration Management

**Old way:**
```python
model = RFDETRBase(
    num_classes=80,
    pretrain_weights="rf-detr-base.pth",
    resolution=640
)
```

**New way:**
```python
from rfdetr.core.config import ConfigurationManager

config_manager = ConfigurationManager()
model_config = config_manager.load_model_config(
    model_name="base",
    num_classes=80,
    resolution=640
)
```

### 2. Model Creation

**Old way:**
```python
# Model creation was hidden inside RFDETR.__init__
model = RFDETRBase()
```

**New way:**
```python
from rfdetr.core.models import ModelFactory

factory = ModelFactory()
model = factory.create_model(model_config)
criterion, postprocessors = factory.create_criterion_and_postprocessors(model_config)
```

### 3. Training

**Old way:**
```python
model = RFDETRBase()
model.train(
    dataset_dir="/path/to/dataset",
    output_dir="/path/to/output",
    epochs=100,
    batch_size=8
)
```

**New way:**
```python
from rfdetr.core.training import Trainer
from rfdetr.core.checkpoint import CheckpointManager

# Create components
checkpoint_manager = CheckpointManager(checkpoint_dir="./checkpoints")
trainer = Trainer(
    model=model,
    criterion=criterion,
    data_loader_factory=data_loader_factory,
    optimizer_factory=optimizer_factory,
    checkpoint_manager=checkpoint_manager
)

# Train
train_config = config_manager.load_training_config(
    dataset_dir="/path/to/dataset",
    output_dir="/path/to/output",
    epochs=100,
    batch_size=8
)
results = trainer.train(train_config)
```

### 4. Callbacks

**Old way:**
```python
# Callbacks were hardcoded in RFDETR.train_from_config
if config.tensorboard:
    # Hardcoded TensorBoard setup
```

**New way:**
```python
from rfdetr.core.training import CallbackManager
from rfdetr.core.training.callbacks import LoggingCallback, ModelCheckpointCallback

callback_manager = CallbackManager()
callback_manager.register_callback_object(LoggingCallback())
callback_manager.register_callback_object(
    ModelCheckpointCallback(checkpoint_manager)
)
```

### 5. Inference

**Old way:**
```python
model = RFDETRBase(pretrain_weights="checkpoint.pth")
detections = model.predict(image, threshold=0.5)
```

**New way:**
```python
from rfdetr.core.inference import Predictor
from rfdetr.core.data import ImagePreprocessor

# Load model
model = factory.create_model(model_config)
factory.load_pretrained_weights(model, "checkpoint.pth")

# Create predictor
preprocessor = ImagePreprocessor(
    means=[0.485, 0.456, 0.406],
    stds=[0.229, 0.224, 0.225],
    resolution=640
)
predictor = Predictor(model, preprocessor, postprocessors)

# Predict
detections = predictor.predict([image], threshold=0.5)
```

## Benefits of Migration

1. **Testability**: Each component can be tested in isolation
2. **Extensibility**: Easy to add new features without modifying core code
3. **Maintainability**: Clear separation of concerns
4. **Flexibility**: Mix and match components as needed
5. **Type Safety**: Full typing support with protocols

## Backward Compatibility

To maintain backward compatibility during migration, we can create wrapper classes:

```python
class RFDETRCompat:
    """Compatibility wrapper for old API."""
    
    def __init__(self, **kwargs):
        self.config_manager = ConfigurationManager()
        self.model_config = self.config_manager.load_model_config(**kwargs)
        self.factory = ModelFactory()
        self.model = self.factory.create_model(self.model_config)
    
    def train(self, **kwargs):
        # Map old API to new components
        train_config = self.config_manager.load_training_config(**kwargs)
        # ... setup trainer and train
    
    def predict(self, images, **kwargs):
        # Map old API to new predictor
        # ... setup predictor and predict
```

## Testing the Migration

Run the test suite to ensure everything works:

```bash
# Test refactored components
python -m pytest tests/test_refactored_components.py -v

# Test backward compatibility
python -m pytest tests/test_compatibility.py -v

# Integration test
python scripts/train.py --steps_per_validation 20 --test_limit 20
```

## Next Steps

1. Gradually migrate existing code to use new components
2. Add more specialized services (MemoryManager, DistributedCoordinator, etc.)
3. Improve error handling and logging
4. Add comprehensive documentation
5. Set up CI/CD pipeline with automated testing