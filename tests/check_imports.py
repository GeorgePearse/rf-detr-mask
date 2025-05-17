import inspect
import sys

import rfdetr
import rfdetr.detr

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print(f"\nrfdetr module location: {rfdetr.__file__}")
print(f"rfdetr.detr module location: {rfdetr.detr.__file__}")

predict_method = rfdetr.detr.RFDETR.predict
predict_source = inspect.getsource(predict_method)
if "return detections_output, masks_output" in predict_source:
    print("\n✗ Old version detected - still returns tuple")
elif "return detections_list" in predict_source:
    print("\n✓ New version detected - returns Detections with masks")
else:
    print("\n? Unknown version")

# Also check if masks are in the detections creation
if "mask=mask_array" in predict_source:
    print("✓ Detections creation includes mask parameter")
else:
    print("✗ Detections creation missing mask parameter")
