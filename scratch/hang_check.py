import torch
from transformers import CvtForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")
model = CvtForImageClassification.from_pretrained("microsoft/cvt-13")
print("Model loaded.")
model.to(device)
print("Model moved.")
