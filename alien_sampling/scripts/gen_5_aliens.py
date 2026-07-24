"""Generate 5 alien images with aggressive MPS cleanup between each."""
import sys
import gc
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.make_images import load_pipeline, build_prompt

pipe = load_pipeline("stabilityai/sdxl-turbo", "mps")

concepts_list = [
    ["impressionism", "forum", "staircase", "hut", "burial", "parliament", "concert", "excavation"],
    ["impressionism", "dull", "hut", "drive", "reservoir", "salmon", "bench", "elbow"],
    ["impressionism", "cap", "supermarket", "gravel", "seal", "cream", "guitar", "lawn"],
    ["impressionism", "coma", "bite", "makeup", "sheep", "sweater", "suspicion"],
    ["impressionism", "flour", "congress", "bond", "flag", "prince", "staircase", "debate"],
]

out_dir = Path("outputs/demo/images_final")
out_dir.mkdir(parents=True, exist_ok=True)
success = 0

for i, concepts in enumerate(concepts_list):
    # Aggressive cleanup before each generation
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()

    prompt = build_prompt(concepts)
    print(f"\n[{i}] Prompt: {prompt[:70]}...")

    seed = 42 + i * 100
    gen = torch.Generator(device="cpu").manual_seed(seed)

    with torch.no_grad():
        result = pipe(
            prompt=prompt, width=512, height=512,
            num_inference_steps=4, guidance_scale=0.0, generator=gen,
        )

    img = result.images[0]
    arr = np.array(img)

    if arr.max() == 0 or arr.std() < 1.0:
        print(f"  CORRUPT (std={arr.std():.2f}), retrying with new seed...")
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()

        gen2 = torch.Generator(device="cpu").manual_seed(seed + 9999)
        with torch.no_grad():
            result = pipe(
                prompt=prompt, width=512, height=512,
                num_inference_steps=4, guidance_scale=0.0, generator=gen2,
            )
        img = result.images[0]
        arr = np.array(img)
        if arr.max() == 0 or arr.std() < 1.0:
            print(f"  STILL CORRUPT, skipping")
            continue

    path = out_dir / f"alien_{i:03d}.png"
    img.save(path)
    success += 1
    print(f"  OK -> {path.name} (std={arr.std():.1f})")

print(f"\nDone: {success}/5 images generated")
