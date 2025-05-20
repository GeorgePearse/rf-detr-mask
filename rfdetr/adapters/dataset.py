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
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
