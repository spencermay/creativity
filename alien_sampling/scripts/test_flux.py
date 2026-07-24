"""Test FLUX image generation via Ollama."""
import ollama
import json

prompt = "A painting that contains the concepts: impressionism lawn bench harvest sunset"

print(f"Prompt: {prompt}")
print("Generating image...")

response = ollama.generate(
    model="x/flux2-klein:4b",
    prompt=prompt,
)

# Inspect response structure
print(f"\nResponse keys: {list(response.keys())}")
print(f"Response type: {type(response)}")

if "images" in response:
    print(f"Number of images: {len(response['images'])}")
    img_data = response["images"][0]
    print(f"Image data type: {type(img_data)}, length: {len(img_data)}")
    # Save it
    import base64
    img_bytes = base64.b64decode(img_data)
    with open("outputs/test_flux.png", "wb") as f:
        f.write(img_bytes)
    print(f"Saved to outputs/test_flux.png ({len(img_bytes)} bytes)")
else:
    # Check other possible fields
    resp_text = response.get("response", "")
    print(f"Response text length: {len(resp_text)}")
    print(f"First 200 chars: {resp_text[:200]}")
    
    # Maybe the image is in the response directly
    if len(resp_text) > 1000:
        import base64
        try:
            img_bytes = base64.b64decode(resp_text)
            with open("outputs/test_flux.png", "wb") as f:
                f.write(img_bytes)
            print(f"Decoded response as image: {len(img_bytes)} bytes -> outputs/test_flux.png")
        except Exception as e:
            print(f"Not base64: {e}")
