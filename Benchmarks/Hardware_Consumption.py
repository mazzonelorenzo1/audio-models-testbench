import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def plot_hwinfo_data(csv_file_path, output_image_path="C:/YourPath/Consumption_Analytics/ModelName.png"):
    print(f"📊 Reading and cleaning HWiNFO log from: {csv_file_path}...")

    # 1. Manual reading to bypass HWiNFO formatting issues
    with open(csv_file_path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')

    # Target columns to search for in the HWiNFO log
    target_keywords = {
        'Time': 'Time',
        'CPU': 'Total CPU Usage [%]',
        'GPU': 'GPU Total Usage [%]',
        'NPU': 'NPU Total Usage [%]',
        'RAM': 'Physical Memory Load [%]'
    }

    # Dynamically find column indices
    indices = {}
    for i, col in enumerate(header):
        clean_col = col.strip('"')
        for key, keyword in target_keywords.items():
            if keyword == clean_col:
                indices[key] = i

    # Extracting data row by row (skipping footers and errors)
    data = {k: [] for k in target_keywords.keys()}

    for line in lines[1:]:
        fields = line.strip().split(',')

        # Ignore empty rows or footer rows
        if not fields or fields[0] == '' or 'Time' in fields[indices['Time']]:
            continue

        try:
            # Check that the time format is valid
            time_val = fields[indices['Time']].strip('"')
            datetime.strptime(time_val, '%H:%M:%S.%f')

            for key in data.keys():
                idx = indices[key]
                if idx < len(fields):
                    data[key].append(fields[idx].strip('"'))
                else:
                    data[key].append(None)
        except Exception:
            continue

    # 2. DataFrame creation and type conversion
    df = pd.DataFrame(data)

    for col in ['CPU', 'GPU', 'NPU', 'RAM']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f')


    print("🎨 Generating the plot...")
    plt.figure(figsize=(12, 6))

    plt.plot(df['Time'], df['CPU'], label='CPU Usage (%)', color='#1f77b4', linewidth=2)
    plt.plot(df['Time'], df['GPU'], label='iGPU Usage (Intel Arc)', color='#2ca02c', linewidth=2)
    plt.plot(df['Time'], df['NPU'], label='NPU Usage (Intel AI Boost)', color='#d62728', linewidth=2)
    plt.plot(df['Time'], df['RAM'], label='System RAM (%)', color='#9467bd', linewidth=2, linestyle='--')

    plt.title('Hardware Resource Allocation (Zero-Overhead Profiling)', fontsize=14, pad=15)
    plt.xlabel('Timeline', fontsize=12)
    plt.ylabel('Utilization / Load (%)', fontsize=12)

    # Fix the Y axis from 0 to 100 to always have the same scale of comparison
    plt.ylim(0, 100)

    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Formatting to show only Time (Hours:Minutes:Seconds)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()

    # Saving in high definition
    plt.savefig(output_image_path, dpi=300)
    print(f"✅ Plot successfully saved to: {output_image_path}")
    plt.show()


# Execute the function on your file
if __name__ == "__main__":
    plot_hwinfo_data("C:/YourPath/CSV_Paper/ModelName.CSV")
