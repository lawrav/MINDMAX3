import numpy as npx
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from scipy.signal import butter, lfilter, welch

def process_aura_data(input_file, output_sheet_name):
    # Step 1: Read the raw data
    df = pd.read_csv(input_file, skiprows=1)  # Skip the units row
    
    # Step 2: Process EEG data to extract frequency bands
    def process_eeg_signals(data, fs=250):  # Assuming 250Hz sampling rate
        # Bandpass filter design
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a
        
        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y
        
        # Frequency bands (in Hz)
        bands = {
            'Theta': (4, 8),
            'Beta': (13, 30)
        }
        
        results = {}
        for band, (low, high) in bands.items():
            filtered = butter_bandpass_filter(data, low, high, fs)
            # Calculate power using Welch's method
            f, Pxx = welch(filtered, fs, nperseg=1024)
            idx = np.logical_and(f >= low, f <= high)
            power = np.trapz(Pxx[idx], f[idx])
            results[f'{band}_Power'] = power
            results[f'{band}_RMS'] = np.sqrt(np.mean(filtered**2))
        
        return results
    
    # Step 3: Process each channel
    channels = ['F3', 'Fz', 'F4', 'C3', 'P3', 'C4', 'Pz', 'P4']
    results = []
    
    for _, row in df.iterrows():
        timestamp = row['Time and date']
        channel_data = {}
        channel_data['Timestamp'] = timestamp
        
        for channel in channels:
            eeg_data = row[channel]
            # Process the signal for this channel
            processed = process_eeg_signals(np.array([eeg_data]))  # Wrap in array for processing
            
            for band in ['Theta', 'Beta']:
                channel_data[f'{channel}_{band}_Power'] = processed[f'{band}_Power']
                channel_data[f'{channel}_{band}_RMS'] = processed[f'{band}_RMS']
        
        results.append(channel_data)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Step 4: Export to Google Sheets
    # Set up credentials
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    
    # You'll need to download the service account credentials JSON file
    creds = ServiceAccountCredentials.from_json_keyfile_name('service_account.json', scope)
    client = gspread.authorize(creds)
    
    # Open or create the sheet
    try:
        sheet = client.open(output_sheet_name).sheet1
    except:
        sheet = client.create(output_sheet_name).sheet1
    
    # Clear existing data and update with new data
    sheet.clear()
    sheet.update([results_df.columns.values.tolist()] + results_df.values.tolist())
    
    print(f"Data successfully processed and uploaded to Google Sheet: {output_sheet_name}")
    return results_df

# Usage example:
# process_aura_data('AURA_RAW___2025-03-10___16;04;11.csv', 'Aura_Beta_Theta_Results')