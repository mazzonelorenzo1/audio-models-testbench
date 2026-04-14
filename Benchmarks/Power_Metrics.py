import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import glob
import csv


def analyze_model_consumption(csv_file, model_name, plot_save_path, tracking_file):
    with open(csv_file, 'r', encoding='latin1') as f:
        lines = f.readlines()

    if not lines:
        print(f"⚠️ The file {csv_file} is empty.")
        return

    # Extract clean headers
    header = [col.strip('"').strip() for col in lines[0].strip().split(',')]

    # Function to find indices
    def find_col(keywords):
        for i, col in enumerate(header):
            if all(kw.lower() in col.lower() for kw in keywords):
                return i
        return None

    # Column mapping
    col_map = {
        'Time': find_col(['Time']),
        'CPU_W': find_col(['CPU Package Power', '[W]']),
        'GPU_W': find_col(['IGPU Power', '[W]']) or find_col(['GPU Power', '[W]']),
        'System_W': find_col(['Total System Power', '[W]']),
        'CPU_Util': find_col(['Total CPU', '[%]']) or find_col(['CPU Usage', '[%]']),
        'GPU_Util': find_col(['GPU Utilization', '[%]']) or find_col(['GPU D3D', '[%]']),
        'RAM_Util': find_col(['Physical Memory Load', '[%]']) or find_col(['Memory Used', '[%]']),
        'NPU_Util': find_col(['NPU Utilization', '[%]']) or find_col(['NPU', '[%]'])
    }

    indices = {k: v for k, v in col_map.items() if v is not None}

    if 'Time' not in indices:
        print(f"⚠️ 'Time' column not found in {model_name}. Skipping file.")
        return

    data = {k: [] for k in indices.keys()}
    for line in lines[1:]:
        fields = line.strip().split(',')
        if not fields or len(fields) <= indices['Time'] or fields[0] == '' or 'Time' in fields[indices['Time']]:
            continue
        try:
            # Timestamp validation
            time_val = fields[indices['Time']].strip('"')
            datetime.strptime(time_val, '%H:%M:%S.%f')
            for k, v in indices.items():
                if v < len(fields):
                    data[k].append(fields[v].strip('"'))
        except:
            continue

    # DataFrame and type conversion
    df = pd.DataFrame(data)
    if df.empty:
        print(f"⚠️ No valid data extracted for {model_name}.")
        return

    metrics = [k for k in df.columns if k != 'Time']
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f')

    # --- 1. TOTAL DURATION CALCULATION (in seconds) ---
    # Subtract the first timestamp from the last
    duration_timedelta = df['Time'].max() - df['Time'].min()
    duration_seconds = duration_timedelta.total_seconds()

    # --- 2. AVERAGES CALCULATION (Watts and Utilization) ---
    averages = {}
    for metric in ['CPU_W', 'GPU_W', 'System_W', 'CPU_Util', 'GPU_Util', 'RAM_Util', 'NPU_Util']:
        averages[metric] = df[metric].mean() if metric in df.columns else 0.0

    # --- 3. ENERGY CALCULATION (Joules) ---
    # Energy (J) = Power (W) * Time (s)
    system_joules = averages['System_W'] * duration_seconds
    cpu_joules = averages['CPU_W'] * duration_seconds
    gpu_joules = averages['GPU_W'] * duration_seconds

    # --- SAVING TO TRACKING FILE ---
    file_exists = os.path.isfile(tracking_file)
    with open(tracking_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Updated header with Joules
            writer.writerow(
                ['Model Name', 'Duration [s]', 'Total System Energy [J]', 'CPU Energy [J]', 'GPU Energy [J]',
                 'Avg CPU [W]', 'Avg GPU [W]', 'Avg System [W]',
                 'Avg CPU Usage [%]', 'Avg GPU Usage [%]', 'Avg RAM Usage [%]', 'Avg NPU Usage [%]'])

        # Writing formatted row
        writer.writerow([
            model_name,
            f"{duration_seconds:.2f}",
            f"{system_joules:.2f}",
            f"{cpu_joules:.2f}",
            f"{gpu_joules:.2f}",
            f"{averages['CPU_W']:.2f}",
            f"{averages['GPU_W']:.2f}",
            f"{averages['System_W']:.2f}",
            f"{averages['CPU_Util']:.2f}",
            f"{averages['GPU_Util']:.2f}",
            f"{averages['RAM_Util']:.2f}",
            f"{averages['NPU_Util']:.2f}"
        ])

    # --- CREATION AND SAVING OF THE PLOT ---
    plt.figure(figsize=(12, 6))
    if 'CPU_W' in df.columns: plt.plot(df['Time'], df['CPU_W'], label='CPU Package Power (W)', color='#d62728',
                                       linewidth=2)
    if 'GPU_W' in df.columns: plt.plot(df['Time'], df['GPU_W'], label='iGPU Power (W)', color='#2ca02c', linewidth=2)
    if 'System_W' in df.columns: plt.plot(df['Time'], df['System_W'], label='Total System Power (W)', color='#1f77b4',
                                          linestyle='--')

    # Adding consumed Joules to the title so you can see them directly in the plot!
    plt.title(
        f'Power Consumption - {model_name}\nDuration: {duration_seconds:.1f}s | Total Energy: {system_joules:.1f} Joules',
        fontsize=14)
    plt.ylabel('Power (Watts)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()

    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Analysis completed for {model_name}: {duration_seconds:.1f}s | {system_joules:.1f} Joules")


def process_all_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure a new file name is used, so it doesn't mix old data
    tracking_csv = os.path.join(output_dir, "Model_Energy_Tracking.csv")

    csv_files = glob.glob(os.path.join(input_dir, "*.csv")) + glob.glob(os.path.join(input_dir, "*.CSV"))
    csv_files = list(set(csv_files))

    if not csv_files:
        print(f"❌ No CSV files found in the folder: {input_dir}")
        return

    print(f"📂 Found {len(csv_files)} files to analyze. Starting Joule calculation...\n")

    for file_path in csv_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        plot_path = os.path.join(output_dir, f"{model_name}_Plot.png")

        analyze_model_consumption(file_path, model_name, plot_path, tracking_csv)

    print(f"\nProcessing finished! The results in JOULES are in the file:\n{tracking_csv}")


# --- FOLDER PATHS ---
INPUT_FOLDER = r"C:\Users\danil\Desktop\Lorenzo\QwenKittenWhisper\CSV_Baseline"
OUTPUT_FOLDER = r"C:\Users\danil\Desktop\Lorenzo\QwenKittenWhisper\Consumption_Analytics_baseline"

# Execution
process_all_files(INPUT_FOLDER, OUTPUT_FOLDER)
