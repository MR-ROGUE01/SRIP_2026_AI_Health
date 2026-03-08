import argparse
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt  # butter creates filter, filtfilt applies it


def butter_bandpass_filter(signal, lowcut=0.17, highcut=0.4, fs=32, order=4):
    # lowcut = minimum breathing frequency (0.17 Hz = 10 breaths/min)
    # highcut = maximum breathing frequency (0.4 Hz = 24 breaths/min)
    # fs = sampling rate (32 Hz for flow/thorac, 4 Hz for spo2)
    # order = sharpness of filter (4 is standard)
    
    nyq = fs / 2  # Nyquist = always half of sampling rate (32/2 = 16 Hz)
    
    low = lowcut / nyq   # convert to 0-1 scale: 0.17/16 = 0.010625
    high = highcut / nyq # convert to 0-1 scale: 0.4/16  = 0.025
    
    # butter() creates the filter recipe - returns two arrays b and a
    # btype='band' means keep only frequencies BETWEEN low and high
    b, a = butter(order, [low, high], btype='band')
    
    # filtfilt applies the filter forward AND backward to avoid distortion
    # input: raw noisy signal → output: clean breathing signal
    filtered = filtfilt(b, a, signal)
    
    return filtered  # return the cleaned signal


def read_signal(filepath):
    rows = []
    in_data = False  # flag to track if we've passed the header section
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()  # remove spaces and \n from start/end
            
            if line == 'Data:':  # this line marks end of header
                in_data = True   # flip flag - start reading data now
                continue         # skip this line itself
            
            if not in_data or line == '':  # skip header lines and blank lines
                continue
            
            parts = line.split(';')  # split "30.05.2024 20:59:00,000; 120" into two parts
            if len(parts) < 2:       # safety check - skip malformed lines
                continue
            
            timestamp = parts[0].strip()  # "30.05.2024 20:59:00,000"
            value = parts[1].strip()      # "120"
            rows.append([timestamp, value])
    
    df = pd.DataFrame(rows, columns=['timestamp', 'value'])
    
    # replace comma with dot in milliseconds: "20:59:00,000" → "20:59:00.000"
    df['timestamp'] = df['timestamp'].str.replace(',', '.')
    
    # convert string to real datetime object so we can do time calculations
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S.%f')
    
    # convert string "120" to actual number 120 so we can do math
    df['value'] = pd.to_numeric(df['value'])
    
    return df


def read_events(filepath):
    df = pd.read_csv(filepath)  # flow_events.csv is already clean so read directly
    df['start_time'] = pd.to_datetime(df['start_time'])  # convert to datetime
    df['end_time'] = pd.to_datetime(df['end_time'])      # convert to datetime
    return df


def create_windows(signal_values, timestamps, window_size=30, overlap=0.5, fs=32):
    # window_size = 30 seconds per window
    # overlap = 0.5 means 50% overlap between consecutive windows
    # fs = sampling rate
    
    # step_size = how many samples to jump forward for each new window
    # 30 * 32 * 0.5 = 480 samples = 15 seconds
    step_size = int(window_size * fs * (1 - overlap))
    
    # total samples in one window: 30 seconds * 32 Hz = 960 samples
    window_samples = int(window_size * fs)
    
    windows = []      # will store signal values for each window
    window_times = [] # will store (start_time, end_time) for each window
    
    # loop through signal jumping step_size samples at a time
    # stop when remaining samples are less than one full window
    for start in range(0, len(signal_values) - window_samples, step_size):
        end = start + window_samples  # end index of this window
        
        window = signal_values[start:end]  # slice 960 samples from signal
        start_time = timestamps[start]     # timestamp at window start
        end_time = timestamps[end - 1]     # timestamp at window end
        
        windows.append(window)                    # add window data
        window_times.append((start_time, end_time))  # add time range
    
    return windows, window_times


def label_window(window_start, window_end, events):
    # check each breathing event against this window
    for _, event in events.iterrows():
        event_start    = event['start_time']
        event_end      = event['end_time']
        
        # overlap starts at whichever is LATER: window start or event start
        overlap_start = max(window_start, event_start)
        
        # overlap ends at whichever is EARLIER: window end or event end
        overlap_end = min(window_end, event_end)
        
        # calculate how many seconds the event and window share
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        
        if overlap_duration <= 0:  # no overlap at all - skip this event
            continue
        
        # window is always 30 seconds
        window_duration = (window_end - window_start).total_seconds()
        
        # if more than 50% of the WINDOW is covered by this event → label it
        if overlap_duration / window_duration > 0.5:
            return event['event_type']
    
    return 'Normal'


def process_participant(participant_folder, events):
    participant_name = os.path.basename(participant_folder)  # "AP01" from "Data/AP01"
    print(f"\nProcessing {participant_name}...")
    
    # Read all 3 signal files
    flow   = read_signal(os.path.join(participant_folder, 'nasal_airflow.txt'))
    thorac = read_signal(os.path.join(participant_folder, 'thoracic_movement.txt'))
    spo2   = read_signal(os.path.join(participant_folder, 'spo2.txt'))
    
    # Filter each signal to keep only breathing frequencies
    # .values converts pandas Series to numpy array (required by filter)
    flow_filtered   = butter_bandpass_filter(flow['value'].values)
    thorac_filtered = butter_bandpass_filter(thorac['value'].values)
    spo2_filtered   = butter_bandpass_filter(spo2['value'].values, fs=4)  # spo2 is 4Hz not 32Hz
    
    # Split each filtered signal into 30 second windows with 50% overlap
    flow_windows,   window_times = create_windows(flow_filtered,   flow['timestamp'].values)
    thorac_windows, _            = create_windows(thorac_filtered, thorac['timestamp'].values)
    spo2_windows,   _            = create_windows(spo2_filtered,   spo2['timestamp'].values, fs=4)
    # _ means we don't need the window_times again - already got them from flow
    
    print(f"  Total windows: {len(flow_windows)}")
    
    # Label each window and build the dataset
    rows = []
    for i in range(len(flow_windows)):
        window_start = pd.Timestamp(window_times[i][0])  # start time of this window
        window_end   = pd.Timestamp(window_times[i][1])  # end time of this window
        
        # check if any breathing event overlaps this window by more than 50%
        label = label_window(window_start, window_end, events)
        
        # store window info as one row in our dataset
        row = {
            'participant': participant_name,
            'window_id':   i,
            'start_time':  window_start,
            'end_time':    window_end,
            'label':       label
        }
        
        # add flow signal values as flow_0, flow_1, ... flow_959
        for j, val in enumerate(flow_windows[i]):
            row[f'flow_{j}'] = val
        
        # add thoracic signal values as thorac_0, thorac_1, ... thorac_959
        for j, val in enumerate(thorac_windows[i]):
            row[f'thorac_{j}'] = val
        
        # add spo2 signal values as spo2_0, spo2_1, ... spo2_119
        for j, val in enumerate(spo2_windows[i]):
            row[f'spo2_{j}'] = val
        
        rows.append(row)
    
    return pd.DataFrame(rows)  # convert list of rows to a pandas DataFrame

def main(in_dir, out_dir):
    all_data = []  # will collect results from all 5 participants
    
    # loop through each participant folder AP01 to AP05
    for participant in sorted(os.listdir(in_dir)):
        participant_folder = os.path.join(in_dir, participant)
        
        # skip if not a folder
        if not os.path.isdir(participant_folder):
            continue
        
        # read this participant's events file
        events_path = os.path.join(participant_folder, 'flow_events.csv')
        events = read_events(events_path)
        
        # process this participant and get their windows as a DataFrame
        df = process_participant(participant_folder, events)
        all_data.append(df)
    
    # combine all participants into one big DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    
    # print summary
    print(f"\nDataset Summary:")
    print(f"Total windows : {len(final_df)}")
    print(f"Label counts  :\n{final_df['label'].value_counts()}")
    
    # save to CSV
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, 'breathing_dataset.csv')
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")


# ── Entry point ──
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create breathing dataset from PSG signals.')
    parser.add_argument('-in_dir',  required=True, help='Folder containing participant data e.g. Data')
    parser.add_argument('-out_dir', required=True, help='Folder to save dataset e.g. Dataset')
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)