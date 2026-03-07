import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

def read_signal(filepath):
    rows = []
    in_data = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == 'Data:':
                in_data = True
                continue
            if not in_data or line == '':
                continue
            parts = line.split(';')
            if len(parts) < 2:
                continue
            timestamp = parts[0].strip()
            value = parts[1].strip()
            rows.append([timestamp, value])
    
    df = pd.DataFrame(rows, columns=['timestamp', 'value'])
    df['timestamp'] = df['timestamp'].str.replace(',', '.')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S.%f')
    df['value'] = pd.to_numeric(df['value'])
    return df


def read_events(filepath):
    df = pd.read_csv(filepath)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    return df


def visualize(participant_folder):
    participant_name = os.path.basename(participant_folder)
    
    # Read all signals
    flow   = read_signal(os.path.join(participant_folder, 'nasal_airflow.txt'))
    thorac = read_signal(os.path.join(participant_folder, 'thoracic_movement.txt'))
    spo2   = read_signal(os.path.join(participant_folder, 'spo2.txt'))
    events = read_events(os.path.join(participant_folder, 'flow_events.csv'))
    
    print(f"Loaded data for {participant_name}")
    print(f"Nasal Airflow  : {len(flow)} samples")
    print(f"Thoracic Move  : {len(thorac)} samples")
    print(f"SpO2           : {len(spo2)} samples")
    print(f"Events         : {len(events)} events")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(f"Sleep Study - {participant_name}", fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=0.08)

    # Plot the 3 signals
    axes[0].plot(flow['timestamp'],   flow['value'],   color='steelblue',  linewidth=0.4)
    axes[1].plot(thorac['timestamp'], thorac['value'], color='darkorange', linewidth=0.4)
    axes[2].plot(spo2['timestamp'],   spo2['value'],   color='green',      linewidth=0.4)

    # Y axis labels
    axes[0].set_ylabel('Nasal Airflow',      fontsize=10, labelpad=10)
    axes[1].set_ylabel('Thoracic\nMovement', fontsize=10, labelpad=10)
    axes[2].set_ylabel('SpO2 (%)',           fontsize=10, labelpad=10)
    axes[2].set_xlabel('Time', fontsize=10)

    # Overlay breathing events as colored shading
    for _, event in events.iterrows():
        start      = event['start_time']
        end        = event['end_time']
        event_type = event['event_type']
        color      = 'orange' if event_type == 'Hypopnea' else 'red'
        for ax in axes:
            ax.axvspan(start, end, alpha=0.3, color=color)

    # Legend
    hypopnea_patch = mpatches.Patch(color='orange', alpha=0.5, label='Hypopnea')
    apnea_patch    = mpatches.Patch(color='red',    alpha=0.5, label='Obstructive Apnea')
    axes[0].legend(handles=[hypopnea_patch, apnea_patch], loc='upper right', fontsize=9)

    # Save as PDF
    os.makedirs('Visualizations', exist_ok=True)
    output_path = os.path.join('Visualizations', f'{participant_name}_visualization.pdf')
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved → {output_path}")


# ── Entry point ──
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise overnight sleep signals.')
    parser.add_argument('-name', required=True, help='Path to participant folder e.g. Data/AP01')
    args = parser.parse_args()
    visualize(args.name)