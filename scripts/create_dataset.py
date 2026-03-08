import argparse
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def butter_bandpass_filter(signal, lowcut=0.17, highcut=0.4, fs=32, order=4):
    # human breathing range is 0.17Hz to 0.4Hz (10-24 breaths/min)
    # anything outside this range is noise
    
    nyq = fs / 2
    low  = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    
    # filtfilt applies filter in both directions to avoid phase distortion
    filtered = filtfilt(b, a, signal)
    
    return filtered


def read_signal(filepath):
    rows = []
    in_data = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # data starts after the 'Data:' line
            if line == 'Data:':
                in_data = True
                continue
            
            if not in_data or line == '':
                continue
            
            # each line is semicolon separated - timestamp;value
            parts = line.split(';')
            if len(parts) < 2:
                continue
            
            timestamp = parts[0].strip()
            value     = parts[1].strip()
            rows.append([timestamp, value])
    
    df = pd.DataFrame(rows, columns=['timestamp', 'value'])
    
    # timestamp uses comma for milliseconds, replacing with dot for parsing
    df['timestamp'] = df['timestamp'].str.replace(',', '.')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S.%f')
    df['value']     = pd.to_numeric(df['value'])
    
    return df


def read_events(filepath):
    df = pd.read_csv(filepath)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time']   = pd.to_datetime(df['end_time'])
    return df


def create_windows(signal_values, timestamps, window_size=30, overlap=0.5, fs=32):
    # 30 sec window with 50% overlap
    # step = 15 seconds = 480 samples at 32Hz
    step_size      = int(window_size * fs * (1 - overlap))
    window_samples = int(window_size * fs)
    
    windows      = []
    window_times = []
    
    for start in range(0, len(signal_values) - window_samples, step_size):
        end        = start + window_samples
        window     = signal_values[start:end]
        start_time = timestamps[start]
        end_time   = timestamps[end - 1]
        
        windows.append(window)
        window_times.append((start_time, end_time))
    
    return windows, window_times


def label_window(window_start, window_end, events):
    for _, event in events.iterrows():
        event_start = event['start_time']
        event_end   = event['end_time']
        
        overlap_start    = max(window_start, event_start)
        overlap_end      = min(window_end, event_end)
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        
        if overlap_duration <= 0:
            continue
        
        window_duration = (window_end - window_start).total_seconds()
        
        # label window with event type if more than 50% overlap
        if overlap_duration / window_duration > 0.5:
            return event['event_type']
    
    return 'Normal'


def process_participant(participant_folder, events):
    participant_name = os.path.basename(participant_folder)
    print(f"\nProcessing {participant_name}...")
    
    flow   = read_signal(os.path.join(participant_folder, 'nasal_airflow.txt'))
    thorac = read_signal(os.path.join(participant_folder, 'thoracic_movement.txt'))
    spo2   = read_signal(os.path.join(participant_folder, 'spo2.txt'))
    
    # apply bandpass filter to remove noise outside breathing frequency range
    flow_filtered   = butter_bandpass_filter(flow['value'].values)
    thorac_filtered = butter_bandpass_filter(thorac['value'].values)
    spo2_filtered   = butter_bandpass_filter(spo2['value'].values, fs=4)
    
    # create 30 second windows with 50% overlap
    flow_windows,   window_times = create_windows(flow_filtered,   flow['timestamp'].values)
    thorac_windows, _            = create_windows(thorac_filtered, thorac['timestamp'].values)
    spo2_windows,   _            = create_windows(spo2_filtered,   spo2['timestamp'].values, fs=4)
    
    print(f"  Total windows: {len(flow_windows)}")
    
    rows = []
    for i in range(len(flow_windows)):
        window_start = pd.Timestamp(window_times[i][0])
        window_end   = pd.Timestamp(window_times[i][1])
        
        label = label_window(window_start, window_end, events)
        
        row = {
            'participant': participant_name,
            'window_id':   i,
            'start_time':  window_start,
            'end_time':    window_end,
            'label':       label
        }
        
        for j, val in enumerate(flow_windows[i]):
            row[f'flow_{j}'] = val
        
        for j, val in enumerate(thorac_windows[i]):
            row[f'thorac_{j}'] = val
        
        for j, val in enumerate(spo2_windows[i]):
            row[f'spo2_{j}'] = val
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main(in_dir, out_dir):
    all_data = []
    
    for participant in sorted(os.listdir(in_dir)):
        participant_folder = os.path.join(in_dir, participant)
        
        if not os.path.isdir(participant_folder):
            continue
        
        events_path = os.path.join(participant_folder, 'flow_events.csv')
        events      = read_events(events_path)
        df          = process_participant(participant_folder, events)
        all_data.append(df)
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nDataset Summary:")
    print(f"Total windows : {len(final_df)}")
    print(f"Label counts  :\n{final_df['label'].value_counts()}")
    
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, 'breathing_dataset.csv')
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create breathing dataset from PSG signals.')
    parser.add_argument('-in_dir',  required=True, help='Folder containing participant data e.g. Data')
    parser.add_argument('-out_dir', required=True, help='Folder to save dataset e.g. Dataset')
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)