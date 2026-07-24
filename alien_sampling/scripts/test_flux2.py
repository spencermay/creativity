"""Test FLUX with minimal prompt."""
import ollama
import base64

print("Attempting FLUX generation...")
try:
    response = ollama.generate(
        model="x/flux2-klein:4b",
        prompt="A painting of a mountain with a castle",
    )
    print(f"Response keys: {list(response.keys())}")
    if "images" in response:
        print(f"Got {len(response['images'])} image(s)")
        img = base64.b64decode(response["images"][0])
        with open("outputs/test_flux.png", "wb") as f:
            f.write(img)
        print(f"Saved: {len(img)} bytes -> outputs/test_flux.png")
    else:
        print(f"Response (first 500 chars): {str(response)[:500]}")
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying with keep_alive=0 to free memory...")
    # Try unloading any models first
    try:
        ollama.generate(model="x/flux2-klein:4b", prompt="", keep_alive=0)
    except:
        pass
