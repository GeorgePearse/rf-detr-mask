# RF-DETR Simple Transforms
# Provides simple transforms that maintain fixed image dimensions

import torch
import torchvision.transforms.functional as F


class FixedResize:
    """Simple transform that resizes images to fixed dimensions."""

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, target=None):
        # Resize the image to the specified dimensions
        resized_img = F.resize(img, (self.height, self.width))

        if target is None:
            return resized_img, None

        # Get original dimensions
        w, h = img.size

        # Calculate scale factors
        h_scale = self.height / h
        w_scale = self.width / w

        # Update target with new size
        target = target.copy()
        target["size"] = torch.tensor([self.height, self.width])

        # Rescale boxes if present
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([w_scale, h_scale, w_scale, h_scale])
            target["boxes"] = scaled_boxes

            # Update area based on scaled boxes
            if "area" in target:
                area = target["area"]
                scaled_area = area * (w_scale * h_scale)
                target["area"] = scaled_area

        # Resize masks if present
        if "masks" in target:
            # Use interpolate for mask resizing
            from rfdetr.util.misc import interpolate

            target["masks"] = (
                interpolate(
                    target["masks"][:, None].float(), (self.height, self.width), mode="nearest"
                )[:, 0]
                > 0.5
            )

        return resized_img, target


class Normalize:
    """Normalize image using given mean and std."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor:
    """Convert PIL image to tensor."""

    def __call__(self, img, target):
        return F.to_tensor(img), target


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def make_simple_transforms(width, height):
    """Create simple transforms that maintain fixed dimensions.

    Args:
        width: Target width
        height: Target height

    Returns:
        A composition of transforms ensuring fixed dimensions
    """
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return Compose(
        [
            ToTensor(),
            FixedResize(height=height, width=width),
            normalize,
        ]
    )
