from transformers import AutoImageProcessor, AutoFeatureExtractor, CvtForImageClassification
import torch

model_name = "microsoft/cvt-13"

try:
    print("Trying AutoFeatureExtractor...")
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    print("Success with AutoFeatureExtractor")
except Exception as e:
    print(f"AutoFeatureExtractor failed: {e}")

try:
    print("Trying AutoImageProcessor...")
    ip = AutoImageProcessor.from_pretrained(model_name)
    print("Success with AutoImageProcessor")
except Exception as e:
    print(f"AutoImageProcessor failed: {e}")

model = CvtForImageClassification.from_pretrained(model_name)
print(f"Model loaded. Config: {model.config}")
