from huggingface_hub import HfApi
import pandas as pd
import time
import os
from collections import defaultdict

def fetch_recent_audio_models():
    api = HfApi()

    tasks = ["text-to-speech", "automatic-speech-recognition", "text-to-audio", "speech-to-speech", "audio-to-audio", "multimodal-audio"]
    keywords = ["tts", "text-to-speech", "stt", "speech-to-text", "asr", "whisper", "speech", "speech-to-speech", "audio-to-audio", "multimodal-audio", "audio", "voices"]

    data = []
    seen_models = set()

    print("🔍 Starting hybrid and quantitative scan (Edge AI < 3GB)...")
    print("⚠️ The size check will take a few minutes. Please wait patiently...\n")

    # --- PHASE 1: Search by Task ---
    for task in tasks:
        models = api.list_models(filter=task, sort="likes", direction=-1, limit=100)
        process_models(api, models, data, seen_models, f"Tag: {task}")

    # --- PHASE 2: Search by Keyword ---
    for kw in keywords:
        models = api.list_models(search=kw, sort="likes", direction=-1, limit=100)
        process_models(api, models, data, seen_models, f"Keyword: {kw}")

    if not data:
        print("\n❌ No models found with these criteria.")
        return

    # DataFrame creation and saving
    df = pd.DataFrame(data)
    df = df.sort_values(by="Downloads (30d)", ascending=False)

    timestamp = time.strftime("%H%M")
    output_filename = f"C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/hf_audio_edge_models_{timestamp}.csv"

    try:
        df.to_csv(output_filename, index=False, sep=';', encoding='utf-8-sig')
        print(f"\n✅ Search completed! Found {len(df)} unique models for Edge AI.")
        print(f"📁 Data saved to: {output_filename}")
        print("\n🏆 Top 5 Edge models:")
        print(df.head().to_string(index=False))
    except PermissionError:
        print(f"\n❌ Error: Close the CSV file in Excel and try again!")


# --- UPDATED HELPER FUNCTION ---
def process_models(api, models, data_list, seen_set, source_label):
    for m in models:
        if m.id in seen_set:
            continue

        likes = getattr(m, "likes", 0)
        if likes is None or likes < 10:
            continue

        try:
            # 1. DOWNLOAD COMPLETE DATA (Real Weights and Dates)
            info = api.model_info(m.id, files_metadata=True)

            # 2. TIME FILTER (Using the REAL date of the repo)
            true_last_mod = getattr(info, "lastModified", getattr(info, "last_modified", None))

            if true_last_mod and hasattr(true_last_mod, "strftime"):
                mod_date_str = true_last_mod.strftime("%Y-%m-%d")
            elif true_last_mod:
                mod_date_str = str(true_last_mod).split("T")[0][:10]
            else:
                mod_date_str = "2000-01-01"

            # If the last real update is before 2024, we discard it
            if mod_date_str < "2024-01-01":
                continue

            # 3. SMART DIMENSIONAL FILTER (Groups by format)
            ext_sizes = {
                'safetensors': 0, 'bin': 0, 'onnx': 0, 'pt': 0, 'tflite': 0
            }

            folder_ext_sizes = defaultdict(lambda: defaultdict(int))

            for f in info.siblings:
                if getattr(f, "size", None) is not None:
                    fname = getattr(f, "rfilename", "")
                    lower_fname = fname.lower()

                    # Identify the extension
                    ext = None
                    if lower_fname.endswith('.safetensors'):
                        ext = 'safetensors'
                    elif lower_fname.endswith('.bin'):
                        ext = 'bin'
                    elif lower_fname.endswith('.onnx'):
                        ext = 'onnx'
                    elif lower_fname.endswith('.pt'):
                        ext = 'pt'
                    elif lower_fname.endswith('.tflite'):
                        ext = 'tflite'

                    if ext:
                        # Extract the name of the folder where the file is located.
                        # If it's in the root folder, it will return an empty string ''
                        folder = os.path.dirname(fname)

                        # Add the weight in its specific folder/extension combination
                        folder_ext_sizes[folder][ext] += f.size

                # Now we look for the heaviest folder+format combination
            max_bytes = 0
            for folder, exts in folder_ext_sizes.items():
                for ext, size in exts.items():
                    if size > max_bytes:
                        max_bytes = size

            size_gb = max_bytes / (1024 ** 3)

            is_end_to_end = "speech-to-speech" in source_label.lower()
            max_allowed_gb = 4.0 if is_end_to_end else 3.0

            # We discard empty repos or those larger than 3 GB
            if size_gb == 0 or size_gb > max_allowed_gb:
                continue

        except Exception as e:
            # If a model gives an error, we ignore it and move to the next one
            continue

        # 4. SAVING CLEAN DATA
        data_list.append({
            "Model": m.id,
            "Search": source_label,
            "Size (GB)": round(size_gb, 2),
            "Downloads (30d)": getattr(m, "downloads", 0),
            "Likes": likes,
            "Last Update": mod_date_str,
            "Link": f"https://huggingface.co/{m.id}"
        })
        seen_set.add(m.id)

if __name__ == "__main__":
    fetch_recent_audio_models()
