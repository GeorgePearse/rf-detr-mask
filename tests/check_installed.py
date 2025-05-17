import inspect
import rfdetr.detr
import rfdetr

print("rfdetr.__file__:", rfdetr.__file__)
print("rfdetr.detr.__file__:", rfdetr.detr.__file__)

# Get the predict method source
predict_method = rfdetr.detr.RFDETR.predict
print("\nPredict method source locations:")
print(inspect.getfile(predict_method))

# Check the code
predict_source = inspect.getsource(predict_method)
print("\nChecking predict method...")
if "return detections_output, masks_output" in predict_source:
    print("✗ Old version detected - still returns tuple")
elif "return detections_list" in predict_source:
    print("✓ New version detected - returns Detections with masks")
else:
    print("? Unknown version")
    
# Also check postprocessors initialization
model_init = rfdetr.main.Model.__init__
init_source = inspect.getsource(model_init)
print("\nChecking Model.__init__...")
if "self.postprocessors" in init_source:
    print("✓ postprocessors initialized in Model")
else:
    print("✗ postprocessors not found in Model")