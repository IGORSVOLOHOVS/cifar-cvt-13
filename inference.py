"""
Inference script for CvT-13 model fine-tuned on CIFAR-10.
"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, CvtForImageClassification
import logging
from rich.console import Console
from rich.panel import Panel

# Setup
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

def run_inference(image_path: str, model_name: str = "microsoft/cvt-13") -> None:
    """Run inference on a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model and Processor
    logger.info(f"Loading model {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = CvtForImageClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # 2. Load and Preprocess Image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 3. Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    
    # CIFAR-10 Classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # The default model is microsoft/cvt-13, which is trained on ImageNet-1k and
    # returns indices up to 999 - indexing the 10-entry list above with one of
    # those raises IndexError. Take the label from the checkpoint itself, and
    # only fall back to the CIFAR-10 names when the checkpoint is a 10-class
    # fine-tune that never got proper id2label entries.
    id2label = getattr(model.config, "id2label", None) or {}
    label = id2label.get(predicted_class_idx)
    if label is None or (len(id2label) == len(classes) and str(label).startswith("LABEL_")):
        label = classes[predicted_class_idx] if predicted_class_idx < len(classes) else str(predicted_class_idx)
    predicted_label = label

    # 4. Results
    console.print(Panel(
        f"[bold green]Prediction:[/bold green] {predicted_label}\n"
        f"[bold blue]Class Index:[/bold blue] {predicted_class_idx}",
        title="Inference Result",
        expand=False
    ))

import argparse

def main() -> None:
    """Entry point for inference CLI."""
    parser = argparse.ArgumentParser(description="Run CvT-13 inference.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--model", type=str, default="microsoft/cvt-13", help="Model name.")
    args = parser.parse_args()

    run_inference(args.image, args.model)

if __name__ == "__main__":
    main()
