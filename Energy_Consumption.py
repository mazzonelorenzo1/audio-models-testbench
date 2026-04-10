import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


# Script for the energy plot
def plot_power_consumption(csv_file):
    with open(csv_file, 'r', encoding='latin1') as f:
        lines = f.readlines()

    header = [col.strip('"') for col in lines[0].strip().split(',')]

    # Energy columns mapping
    target_cols = {
        'Time': 'Time',
        'CPU_W': 'CPU Package Power [W]',
        'GPU_W': 'IGPU Power [W]',
        'System_W': 'Total System Power [W]'
    }

    indices = {k: header.index(v) for k, v in target_cols.items() if v in header}

    data = {k: [] for k in indices.keys()}
    for line in lines[1:]:
        fields = line.strip().split(',')
        if not fields or fields[0] == '' or 'Time' in fields[indices['Time']]: continue
        try:
            time_val = fields[indices['Time']].strip('"')
            datetime.strptime(time_val, '%H:%M:%S.%f')
            for k, v in indices.items(): data[k].append(fields[v].strip('"'))
        except:
            continue

    df = pd.DataFrame(data)
    for col in ['CPU_W', 'GPU_W', 'System_W']: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f')

    plt.figure(figsize=(12, 6))
    plt.plot(df['Time'], df['CPU_W'], label='CPU Package Power (W)', color='#d62728', linewidth=2)
    plt.plot(df['Time'], df['GPU_W'], label='iGPU Power (W)', color='#2ca02c', linewidth=2)
    plt.plot(df['Time'], df['System_W'], label='Total System Power (W)', color='#1f77b4', linestyle='--')

    plt.title('Power Consumption Profile (Watts) - Khadas Mind 2', fontsize=14)
    plt.ylabel('Power (Watts)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    plt.savefig('C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Consumption_Analytics/WhisperTinyGPU.png', dpi=300)
    plt.show()


plot_power_consumption("C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/CSV_Paper/whisper_tiny_gpu.CSV")
