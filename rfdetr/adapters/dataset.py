import torchvision
import torch
import pycocotools.mask as mask_util
import numpy as np

class ConvertCoco:
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Handle segmentation masks if available
        masks = None
        if anno and "segmentation" in anno[0]:
            masks = []
            for obj in anno:
                if "segmentation" in obj:
                    if isinstance(obj["segmentation"], list):
                        # Polygon format
                        rles = mask_util.frPyObjects(obj["segmentation"], h, w)
                        rle = mask_util.merge(rles)
                    else:
                        # RLE format
                        rle = obj["segmentation"]
                    mask = mask_util.decode(rle)
                    masks.append(mask)
                else:
                    # Create an empty mask if segmentation is missing
                    masks.append(np.zeros((h, w), dtype=np.uint8))

            if masks:
                masks = torch.as_tensor(np.stack(masks), dtype=torch.bool)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if masks is not None:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj.get("iscrowd", 0) for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # Add masks to target if available
        if masks is not None:
            target["masks"] = masks

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, test_limit=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCoco()

        # Limit dataset size for testing if specified
        if test_limit is not None and test_limit > 0:
            self.ids = self.ids[: min(test_limit, len(self.ids))]

    def __getitem__(self, idx):
        try:
            img, target = super().__getitem__(idx)
            image_id = self.ids[idx]
            target = {"image_id": image_id, "annotations": target}
            img, target = self.prepare(img, target)
            
            # Skip transforms and return minimal output for empty or problematic samples
            if len(target["boxes"]) == 0 or torch.any(target["boxes"][:, 0] >= target["boxes"][:, 2]) or torch.any(target["boxes"][:, 1] >= target["boxes"][:, 3]):
                # Convert image to tensor manually
                img_np = np.array(img)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                # Normalize using ImageNet stats
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                
                # Create empty target
                h, w = img_np.shape[:2]
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros(0, dtype=torch.int64)
                if "masks" in target:
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.bool)
                if "area" in target:
                    target["area"] = torch.zeros(0, dtype=torch.float32)
                if "iscrowd" in target:
                    target["iscrowd"] = torch.zeros(0, dtype=torch.uint8)
                target["size"] = torch.tensor([h, w])
                target["orig_size"] = torch.tensor([h, w])
                
                return img_tensor, target
            
            # Apply transforms to valid samples
            if self._transforms is not None:
                # Convert from PIL image to numpy for albumentations
                img_np = np.array(img)
                
                # Just transform the image, we'll handle boxes separately
                transformed = self._transforms(image=img_np)
                transformed_img = transformed["image"]  # This is already a tensor from ToTensorV2
                
                # Resize boxes manually
                src_w, src_h = img.size
                dst_h, dst_w = transformed_img.shape[1:3]  # PyTorch image: (C, H, W)
                
                # Calculate scaling factors
                w_ratio = dst_w / src_w
                h_ratio = dst_h / src_h
                
                # Scale boxes
                boxes = target["boxes"].clone()
                boxes[:, 0] *= w_ratio  # x_min
                boxes[:, 1] *= h_ratio  # y_min
                boxes[:, 2] *= w_ratio  # x_max
                boxes[:, 3] *= h_ratio  # y_max
                
                # Update target
                target["boxes"] = boxes
                # Update image size information
                target["size"] = torch.tensor([dst_h, dst_w])
                
                return transformed_img, target
            
            return img, target
        except Exception as e:
            # Fallback for any errors - return a valid but empty sample
            # This ensures the dataloader doesn't crash during training
            print(f"Error processing sample {idx}: {str(e)}")
            
            # Create a dummy image of the required size (e.g., target size)
            h, w = 672, 560  # Use the target size from configuration
            img_tensor = torch.zeros((3, h, w), dtype=torch.float32)
            
            # Create an empty target dictionary
            dummy_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([0]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.uint8),
                "orig_size": torch.tensor([h, w]),
                "size": torch.tensor([h, w])
            }
            
            return img_tensor, dummy_target
