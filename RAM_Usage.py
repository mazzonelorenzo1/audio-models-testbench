import pandas as pd
from datetime import datetime
import os
import glob
import csv


def analyze_peak_memory(csv_file, model_name, tracking_file):
    with open(csv_file, 'r', encoding='latin1') as f:
        lines = f.readlines()

    if not lines: return
    header = [col.strip('"').strip() for col in lines[0].strip().split(',')]

    def find_col(keywords):
        for i, col in enumerate(header):
            if all(kw.lower() in col.lower() for kw in keywords): return i
        return None

    # We only look for Time and RAM to make it fast
    col_map = {
        'Time': find_col(['Time']),
        'RAM_Util': find_col(['Physical Memory Load', '[%]']) or find_col(['Memory Used', '[%]'])
    }
    indices = {k: v for k, v in col_map.items() if v is not None}

    if 'Time' not in indices or 'RAM_Util' not in indices: return

    data = {'RAM_Util': []}
    for line in lines[1:]:
        fields = line.strip().split(',')
        if not fields or len(fields) <= indices['Time'] or fields[0] == '' or 'Time' in fields[
            indices['Time']]: continue
        try:
            if indices['RAM_Util'] < len(fields):
                data['RAM_Util'].append(fields[indices['RAM_Util']].strip('"'))
        except:
            continue

    df = pd.DataFrame(data)
    df['RAM_Util'] = pd.to_numeric(df['RAM_Util'], errors='coerce')

    # CALCULATING THE MAXIMUM PEAK
    peak_ram = df['RAM_Util'].max() if not df.empty else 0.0

    # Save to file
    file_exists = os.path.isfile(tracking_file)
    with open(tracking_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Model Name', 'Peak RAM Usage [%]'])

        writer.writerow([model_name, f"{peak_ram:.2f}"])

    print(f"✅ {model_name} -> RAM Peak: {peak_ram:.2f}%")


def process_all_files(input_dir, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    tracking_csv = os.path.join(output_dir, "Model_Peak_RAM_Tracking.csv")

    csv_files = glob.glob(os.path.join(input_dir, "*.csv")) + glob.glob(os.path.join(input_dir, "*.CSV"))
    csv_files = list(set(csv_files))

    print(f"📂 Extracting Peak RAM...\n")
    for file_path in csv_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        analyze_peak_memory(file_path, model_name, tracking_csv)

    print(f"\nDone! The maximum peaks are in:\n{tracking_csv}")


# --- FOLDER PATHS ---
INPUT_FOLDER = r"C:\Users\danil\Desktop\Lorenzo\QwenKittenWhisper\CSV_Paper_TTS"
OUTPUT_FOLDER = r"C:\Users\danil\Desktop\Lorenzo\QwenKittenWhisper\Consumption_Analytics"

process_all_files(INPUT_FOLDER, OUTPUT_FOLDER)
