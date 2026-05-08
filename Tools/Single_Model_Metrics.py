from huggingface_hub import HfApi


def inspect_single_model(model_id):
    api = HfApi()

    print(f"🔍 Deep inspection of the model: '{model_id}'...\n")

    try:
        info = api.model_info(model_id, files_metadata=True)
    except Exception as e:
        print(f"❌ API Error: Impossible to find the model. Does it really exist? ({e})")
        return

    # --- 1. GENERAL AND QUALITATIVE METRICS ---
    print("=== 📊 GENERAL METRICS ===")
    print(f"Author:             {getattr(info, 'author', 'Unknown')}")
    print(f"Downloads (30d):    {getattr(info, 'downloads', 0)}")
    print(f"Likes:              {getattr(info, 'likes', 0)}")

    true_last_mod = getattr(info, "lastModified", getattr(info, "last_modified", None))
    if true_last_mod and hasattr(true_last_mod, "strftime"):
        mod_date_str = true_last_mod.strftime("%Y-%m-%d")
    elif true_last_mod:
        mod_date_str = str(true_last_mod).split("T")[0][:10]
    else:
        mod_date_str = "Unknown"

    print(f"Last Update:        {mod_date_str}")
    tags = getattr(info, 'tags', [])
    print(f"Official Tags:      {', '.join(tags) if tags else 'No tags!'}")

    print("\n=== ⚖️ WEIGHT ANALYSIS (Memory Footprint) ===")

    ext_sizes = {
        'safetensors': 0, 'bin': 0, 'onnx': 0, 'pt': 0, 'tflite': 0
    }

    files_found = False

    # Go through every file inside the repository
    for f in info.siblings:
        if getattr(f, "size", None) is not None:
            fname = getattr(f, "rfilename", "")
            lower_fname = fname.lower()

            # Filter only the neural files (weights)
            if lower_fname.endswith(('.safetensors', '.bin', '.onnx', '.pt', '.tflite')):
                files_found = True
                size_mb = f.size / (1024 ** 2)
                print(f" 📄 File: {fname} -> {size_mb:.2f} MB")

                # Add to the specific format counter
                if lower_fname.endswith('.safetensors'):
                    ext_sizes['safetensors'] += f.size
                elif lower_fname.endswith('.bin'):
                    ext_sizes['bin'] += f.size
                elif lower_fname.endswith('.onnx'):
                    ext_sizes['onnx'] += f.size
                elif lower_fname.endswith('.pt'):
                    ext_sizes['pt'] += f.size
                elif lower_fname.endswith('.tflite'):
                    ext_sizes['tflite'] += f.size

    if not files_found:
        print(" ⚠️ No standard neural file found in this repo!")

    print("\n=== 🧮 FINAL CALCULATION FOR EDGE AI ===")

    # Show how much the various blocks weigh
    for ext, size in ext_sizes.items():
        if size > 0:
            print(f" - Total .{ext} format: {size / (1024 ** 3):.2f} GB")

    real_bytes = max(ext_sizes.values()) if ext_sizes else 0
    size_gb = real_bytes / (1024 ** 3)

    print("-" * 40)
    print(f"🚀 ESTIMATED MODEL WEIGHT: {size_gb:.2f} GB")

    if size_gb == 0:
        print("❌ VERDICT: The model would be discarded (Zero weight).")
    elif size_gb > 3.0:
        print("❌ VERDICT: The model would be discarded (> 3.0 GB, too heavy for Edge).")
    else:
        print("✅ VERDICT: The model would pass the dimensional filter!")


if __name__ == "__main__":
    # Insert the exact model name here (e.g. "openai/whisper-tiny" or "LiquidAI/LFM2-700M")
    MODEL_ID = ("ModelRepo/ModelName")

    inspect_single_model(MODEL_ID)
