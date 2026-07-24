from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(text=["a painting of castle"], images=None, return_tensors="pt", padding=True)
print("Input keys:", list(inputs.keys()))

# Remove pixel_values if present (no images)
text_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
print("Text input keys:", list(text_inputs.keys()))

out = model.get_text_features(**text_inputs)
print("Output type:", type(out))
print("Output shape:", out.shape if hasattr(out, "shape") else "no shape attr")
