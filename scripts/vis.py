import argparse                                    # reads command line arguments like -name "Data/AP01"
import os                                          # file/folder operations like path joining, makedirs
import numpy as np                                 # numerical operations on arrays
import pandas as pd                                # reading files and handling timestamps
import matplotlib.pyplot as plt                    # plotting the signals
from matplotlib.backends.backend_pdf import PdfPages  # saving plot as PDF file
import matplotlib.patches as mpatches             # creating colored boxes for legend


def read_signal(filepath):
    rows = []
    in_data = False  # flag - False means we're still in header, True means we reached data
    
    with open(filepath, 'r') as f:  # open file in read mode
        for line in f:              # read one line at a time
            line = line.strip()     # remove spaces and \n from start and end
            
            if line == 'Data:':     # this line marks where actual data begins
                in_data = True      # flip flag to True - start reading data
                continue            # skip this line itself, go to next line
            
            if not in_data or line == '':  # skip all header lines and blank lines
                continue
            
            parts = line.split(';')  # split "30.05.2024 20:59:00,000; 120" into two parts
            if len(parts) < 2:       # safety check - if line doesn't have ; skip it
                continue
            
            timestamp = parts[0].strip()  # first part = "30.05.2024 20:59:00,000"
            value = parts[1].strip()      # second part = "120"
            rows.append([timestamp, value])  # add as one row to our list
    
    # convert list of rows into a pandas DataFrame with proper column names
    df = pd.DataFrame(rows, columns=['timestamp', 'value'])
    
    # replace comma with dot in milliseconds so Python can parse it
    # "20:59:00,000" → "20:59:00.000"
    df['timestamp'] = df['timestamp'].str.replace(',', '.')
    
    # convert string "30.05.2024 20:59:00.000" to real datetime object
    # so we can use it on x-axis and do time calculations
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S.%f')
    
    # convert string "120" to actual number 120 so we can plot it
    df['value'] = pd.to_numeric(df['value'])
    
    return df  # return clean DataFrame with 2 columns: timestamp and value


def read_events(filepath):
    # flow_events.csv is already clean so pandas can read it directly
    df = pd.read_csv(filepath)
    
    # convert start and end times to datetime objects for plotting
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    return df  # return DataFrame with columns: start_time, end_time, duration, event_type, stage


def visualize(participant_folder):
    # extract just "AP01" from "Data/AP01" for use in title and filename
    participant_name = os.path.basename(participant_folder)
    
    # Read all 4 files we need
    flow   = read_signal(os.path.join(participant_folder, 'nasal_airflow.txt'))
    thorac = read_signal(os.path.join(participant_folder, 'thoracic_movement.txt'))
    spo2   = read_signal(os.path.join(participant_folder, 'spo2.txt'))
    events = read_events(os.path.join(participant_folder, 'flow_events.csv'))
    
    # print summary to confirm files loaded correctly
    print(f"Loaded data for {participant_name}")
    print(f"Nasal Airflow  : {len(flow)} samples")
    print(f"Thoracic Move  : {len(thorac)} samples")
    print(f"SpO2           : {len(spo2)} samples")
    print(f"Events         : {len(events)} events")

    # create one figure with 3 subplots stacked vertically
    # sharex=True means all 3 plots share the same x-axis (time)
    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    
    # add main title at top of entire figure
    fig.suptitle(f"Sleep Study - {participant_name}", fontsize=14, fontweight='bold')
    
    # add more space on left so y-axis labels are not cut off
    fig.subplots_adjust(left=0.08)

    # plot each signal on its own subplot
    # linewidth=0.4 because we have 875184 points - thin lines look better
    axes[0].plot(flow['timestamp'],   flow['value'],   color='steelblue',  linewidth=0.4)
    axes[1].plot(thorac['timestamp'], thorac['value'], color='darkorange', linewidth=0.4)
    axes[2].plot(spo2['timestamp'],   spo2['value'],   color='green',      linewidth=0.4)

    # add labels to y-axis of each subplot
    # labelpad=10 adds space between label and plot
    axes[0].set_ylabel('Nasal Airflow',      fontsize=10, labelpad=10)
    axes[1].set_ylabel('Thoracic\nMovement', fontsize=10, labelpad=10)  # \n = new line
    axes[2].set_ylabel('SpO2 (%)',           fontsize=10, labelpad=10)
    axes[2].set_xlabel('Time', fontsize=10)  # x-axis label only on bottom plot

    # loop through each breathing event and draw colored shading
    for _, event in events.iterrows():  # _ = row index (we don't need it)
        start      = event['start_time']
        end        = event['end_time']
        event_type = event['event_type']
        
        # orange for Hypopnea, red for Obstructive Apnea
        color = 'orange' if event_type == 'Hypopnea' else 'red'
        
        # draw shading on ALL 3 subplots for this event
        # alpha=0.3 means 30% opacity so signal is still visible behind shading
        for ax in axes:
            ax.axvspan(start, end, alpha=0.3, color=color)

    # create colored boxes for the legend
    hypopnea_patch = mpatches.Patch(color='orange', alpha=0.5, label='Hypopnea')
    apnea_patch    = mpatches.Patch(color='red',    alpha=0.5, label='Obstructive Apnea')
    
    # add legend to top right of first subplot
    axes[0].legend(handles=[hypopnea_patch, apnea_patch], loc='upper right', fontsize=9)

    # create Visualizations folder if it doesn't exist
    # exist_ok=True means don't crash if folder already exists
    os.makedirs('Visualizations', exist_ok=True)
    
    # build output path e.g. "Visualizations/AP01_visualization.pdf"
    output_path = os.path.join('Visualizations', f'{participant_name}_visualization.pdf')
    
    # save figure as PDF
    # bbox_inches='tight' removes extra white space around the plot
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)  # close figure to free memory - important when processing multiple participants
    
    print(f"Saved → {output_path}")


# ── Entry point ──
# if __name__ == '__main__' means:
# only run this block when script is run directly from command line
# not when it is imported by another script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise overnight sleep signals.')
    
    # add -name argument - required means script will error if not provided
    parser.add_argument('-name', required=True, help='Path to participant folder e.g. Data/AP01')
    
    args = parser.parse_args()  # read the arguments from command line
    
    visualize(args.name)  # args.name contains whatever was passed after -name