import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import simpson as simps
import os
from datetime import datetime
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eeg_processing.log")
    ]
)
logger = logging.getLogger(__name__)

# Set matplotlib backend to 'Agg' for non-interactive environments
import matplotlib
matplotlib.use('Agg')

# Set style for better-looking plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid", font_scale=1.2)
sns.set_palette("viridis")

# Define frequency bands globally
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 45)
}

# Define EEG channel mapping globally
CHANNEL_MAP = {
    'F3': 'uV',
    'Fz': 'uV.1',
    'F4': 'uV.2',
    'C3': 'uV.3',
    'P3': 'uV.4',
    'C4': 'uV.5',
    'Pz': 'uV.6',
    'P4': 'uV.7'
}

def find_folder(target_folder_name, start_path=None):
    """
    Recursively search for a folder with the given name starting from start_path
    
    Args:
        target_folder_name: Name of the folder to find
        start_path: Path to start searching from (default is current directory)
    
    Returns:
        Path to the found folder or None if not found
    """
    if start_path is None:
        # Start from common locations
        potential_start_paths = [
            os.path.expanduser("~"),  # Home directory
            "/",                      # Root directory (Unix)
            "C:\\",                   # Root directory (Windows)
            os.getcwd(),              # Current working directory
        ]
    else:
        potential_start_paths = [start_path]
    
    # Try the specific path first if provided
    specific_path = "/Users/lawra/MINDLAND3/MINDMAX_PROJECT/Users_StressRatio"
    if os.path.exists(specific_path) and os.path.isdir(specific_path):
        logger.info(f"Found target folder at specific path: {specific_path}")
        return specific_path
        
    for start_path in potential_start_paths:
        if not os.path.exists(start_path):
            continue
            
        logger.info(f"Searching for {target_folder_name} in {start_path}")
        
        for root, dirs, _ in os.walk(start_path):
            if target_folder_name in dirs:
                found_path = os.path.join(root, target_folder_name)
                logger.info(f"Found target folder: {found_path}")
                return found_path
                
            # Skip certain directories to improve search speed
            if '.git' in dirs:
                dirs.remove('.git')
            if 'node_modules' in dirs:
                dirs.remove('node_modules')
            if '.venv' in dirs:
                dirs.remove('.venv')
                
    logger.warning(f"Could not find folder {target_folder_name}")
    return None

def process_eeg_data(csv_file, output_dir=None):
    """
    Process EEG data to extract frequency bands and stress metrics
    
    Args:
        csv_file: Path to CSV file with EEG data
        output_dir: Directory to save results (optional)
    
    Returns:
        Tuple with DataFrame of results, DataFrame of transformed raw data, and PSD data dictionary
    """
    logger.info(f"Processing file: {csv_file}")
    
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Load data, skipping the units row
        df = pd.read_csv(csv_file, skiprows=1, quotechar="'")
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_file}: {str(e)}")
        raise
    
    # Parameters
    fs = 250  # Sampling frequency (Hz)
    window_size = 4 * fs  # 4-second windows
    
    results = []
    raw_data_dict = {}
    psd_data_dict = {}
    
    # Process each channel
    channels = ['F3', 'Fz', 'F4', 'C3', 'P3', 'C4', 'Pz', 'P4']
    
    # Save raw data for plotting
    for channel in channels:
        col_name = CHANNEL_MAP[channel]
        if col_name in df.columns:
            try:
                raw_data_dict[channel] = df[col_name].astype(float).values
            except ValueError:
                logger.warning(f"Could not convert column {col_name} to float")
                continue
        else:
            logger.warning(f"Column {col_name} not found for channel {channel}")
            logger.debug(f"Available columns: {df.columns.tolist()}")
            continue
    
    for channel in channels:
        # Get signal using mapped column name
        if CHANNEL_MAP[channel] not in df.columns:
            continue
            
        try:
            signal = df[CHANNEL_MAP[channel]].values.astype(float)
            
            # Calculate PSD using Welch's method
            freqs, psd = welch(signal, fs=fs, nperseg=window_size)
            
            # Save PSD data for plotting
            psd_data_dict[channel] = {'freqs': freqs, 'psd': psd}
            
            # Calculate absolute power in each band
            band_powers = {}
            total_power = simps(psd, freqs)
            
            for band, (low, high) in BANDS.items():
                # Find frequency indices in band
                idx = np.logical_and(freqs >= low, freqs <= high)
                
                # Calculate absolute power by integrating PSD
                band_power = simps(psd[idx], freqs[idx])
                band_powers[band] = band_power
            
            # Calculate relative powers (stress metrics)
            beta_relative = band_powers['Beta'] / total_power if total_power > 0 else 0
            theta_relative = band_powers['Theta'] / total_power if total_power > 0 else 0
            alpha_relative = band_powers['Alpha'] / total_power if total_power > 0 else 0
            
            # Calculate stress ratio (common metric)
            # We use theta/beta as inverse stress indicator (higher values = less stress)
            stress_ratio = theta_relative / beta_relative if beta_relative > 0 else 0
            
            # Calculate frontal alpha asymmetry (emotional valence indicator)
            # Only applies to F3 and F4 channels
            frontal_alpha_asymmetry = 0
            if channel == 'F3':
                f3_alpha = band_powers['Alpha']
                if 'F4' in CHANNEL_MAP and CHANNEL_MAP['F4'] in df.columns:
                    # Save for use when processing F4
                    raw_data_dict['F3_alpha'] = f3_alpha
            elif channel == 'F4':
                f4_alpha = band_powers['Alpha']
                if 'F3_alpha' in raw_data_dict:
                    # Only calculate if F3 was processed
                    frontal_alpha_asymmetry = np.log(f4_alpha / raw_data_dict['F3_alpha']) if raw_data_dict['F3_alpha'] > 0 else 0
            
            results.append({
                'Channel': channel,
                'Delta': band_powers['Delta'],
                'Theta': band_powers['Theta'],
                'Alpha': band_powers['Alpha'],
                'Beta': band_powers['Beta'],
                'Gamma': band_powers['Gamma'],
                'Beta_Relative': beta_relative,
                'Theta_Relative': theta_relative,
                'Alpha_Relative': alpha_relative,
                'Stress_Ratio': stress_ratio,
                'Frontal_Alpha_Asymmetry': frontal_alpha_asymmetry
            })
        except Exception as e:
            logger.error(f"Error processing channel {channel}: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    
    # Convert raw data to DataFrame for analysis and plots
    if raw_data_dict:
        try:
            # Get only numpy arrays
            array_data = {k: v for k, v in raw_data_dict.items() if isinstance(v, np.ndarray)}
            if array_data:
                max_length = max(len(val) for val in array_data.values())
                
                # Process and pad arrays
                raw_data_processed = {}
                for channel, data in array_data.items():
                    if len(data) < max_length:
                        # Pad with NaN if needed
                        raw_data_processed[channel] = np.pad(
                            data, 
                            (0, max_length - len(data)), 
                            mode='constant', 
                            constant_values=np.nan
                        )
                    else:
                        raw_data_processed[channel] = data
                
                raw_df = pd.DataFrame(raw_data_processed)
            else:
                raw_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error creating raw DataFrame: {str(e)}")
            raw_df = pd.DataFrame()
    else:
        raw_df = pd.DataFrame()
    
    return results_df, raw_df, psd_data_dict

def analyze_stress_metrics(results_df, user_id=None):
    """
    Analyze stress metrics and propose thresholds
    
    Args:
        results_df: DataFrame with processed results
        user_id: User identifier (optional)
    
    Returns:
        Dict with analyses and recommended thresholds
    """
    stress_analysis = {}
    
    if results_df.empty:
        logger.warning("No data available for stress analysis")
        return {
            'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0,
            'frontal_mean': 0, 'parietal_mean': 0,
            'suggested_threshold': 0, 'overall_stress_level': "UNKNOWN"
        }
    
    # Extract stress metrics for all channels
    stress_ratios = results_df['Stress_Ratio'].values
    
    # Basic statistics
    stress_min = np.min(stress_ratios)
    stress_max = np.max(stress_ratios)
    stress_mean = np.mean(stress_ratios)
    stress_median = np.median(stress_ratios)
    stress_std = np.std(stress_ratios)
    
    # Analyze by channel group (frontal vs parietal)
    frontal_mask = results_df['Channel'].isin(['F3', 'Fz', 'F4'])
    parietal_mask = results_df['Channel'].isin(['P3', 'Pz', 'P4'])
    
    frontal_stress = results_df.loc[frontal_mask, 'Stress_Ratio'].mean() if any(frontal_mask) else 0
    parietal_stress = results_df.loc[parietal_mask, 'Stress_Ratio'].mean() if any(parietal_mask) else 0
    
    # Propose threshold based on statistics
    # Typically, lower values indicate higher stress for theta/beta ratio
    suggested_threshold = stress_mean - (0.5 * stress_std)
    
    # Evaluate general stress level
    overall_stress_level = "HIGH" if stress_mean < suggested_threshold else "NORMAL"
    
    stress_analysis = {
        'min': stress_min,
        'max': stress_max,
        'mean': stress_mean,
        'median': stress_median,
        'std': stress_std,
        'frontal_mean': frontal_stress,
        'parietal_mean': parietal_stress,
        'suggested_threshold': suggested_threshold,
        'overall_stress_level': overall_stress_level
    }
    
    # Add user ID if provided
    if user_id:
        stress_analysis['User'] = user_id
    
    return stress_analysis

def generate_big_graph(summary_df, output_dir):
    """
    Generate a comprehensive BigGraph showing stress ratio comparison between all users
    
    Args:
        summary_df: DataFrame with stress analysis for multiple users
        output_dir: Directory to save the graph
    """
    try:
        if summary_df.empty or 'User' not in summary_df.columns:
            logger.warning("No user data available for BigGraph")
            return
        
        # Create figure with increased resolution and quality
        plt.figure(figsize=(20, 12), dpi=300)
        
        # Sort users by stress ratio to see patterns better
        sorted_df = summary_df.sort_values(by='mean', ascending=True)
        
        # Create color palette based on stress levels
        colors = []
        for _, row in sorted_df.iterrows():
            if row['mean'] < row['suggested_threshold']:
                colors.append('#FF5555')  # Red for high stress
            else:
                colors.append('#4CAF50')  # Green for normal stress
        
        # Create enhanced bar chart
        ax = sns.barplot(x='User', y='mean', data=sorted_df, palette=colors)
        
        # Add horizontal threshold lines for each user
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            plt.hlines(y=row['suggested_threshold'], xmin=i-0.4, xmax=i+0.4, 
                      colors='black', linestyles='dashed', alpha=0.7, linewidth=1.5)
        
        # Add average threshold line
        avg_threshold = sorted_df['suggested_threshold'].mean()
        plt.axhline(y=avg_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Average Threshold: {avg_threshold:.4f}')
        
        # Add value labels on top of bars
        for i, v in enumerate(sorted_df['mean']):
            ax.text(i, v + 0.03, f"{v:.3f}", ha='center', va='bottom', fontsize=11, 
                   fontweight='bold', color='black')
        
        # Add stress level indicators
        for i, row in enumerate(sorted_df.itertuples()):
            stress_level = "HIGH STRESS" if row.mean < row.suggested_threshold else "NORMAL"
            color = 'white'
            ax.text(i, row.mean/2, stress_level, ha='center', fontsize=12, 
                   color=color, weight='bold')
        
        # Add statistics in the plot
        stats_text = (
            f"Group Statistics:\n"
            f"Average Stress Ratio: {summary_df['mean'].mean():.4f}\n"
            f"Median Stress Ratio: {summary_df['mean'].median():.4f}\n"
            f"High Stress Count: {sum(summary_df['mean'] < summary_df['suggested_threshold'])}"
        )
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Enhance appearance
        plt.title('Comprehensive Stress Ratio Comparison Between All Participants', fontsize=20, pad=20)
        plt.ylabel('Theta/Beta Ratio (Mean)', fontsize=16)
        plt.xlabel('Participant', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        # Add border
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        
        # Save with tight layout
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'BigGraph_User_Stress_Comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"BigGraph saved to: {output_path}")
        print("✅ Tu gráfica vertical ha sido generada correctamente.")
        
        # Create a horizontal version for better visibility with many users
        plt.figure(figsize=(16, max(10, len(sorted_df) * 0.7)), dpi=300)
        
        # Horizontal bar chart
        ax = sns.barplot(y='User', x='mean', data=sorted_df, palette=colors, orient='h')
        
        # Add vertical threshold lines for each user
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            plt.vlines(x=row['suggested_threshold'], ymin=i-0.4, ymax=i+0.4, 
                      colors='black', linestyles='dashed', alpha=0.7, linewidth=1.5)
        
        # Add average threshold line
        plt.axvline(x=avg_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Average Threshold: {avg_threshold:.4f}')
        
        # Add value labels on bars
        for i, v in enumerate(sorted_df['mean']):
            ax.text(v + 0.03, i, f"{v:.3f}", va='center', fontsize=11, 
                   fontweight='bold', color='black')
        
        # Add stress level indicators
        for i, row in enumerate(sorted_df.itertuples()):
            stress_level = "HIGH STRESS" if row.mean < row.suggested_threshold else "NORMAL"
            color = 'white'
            ax.text(row.mean/2, i, stress_level, va='center', fontsize=11, 
                   color=color, weight='bold')
        
        # Add statistics in the plot
        plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Enhance appearance
        plt.title('Comprehensive Stress Ratio Comparison (Horizontal View)', fontsize=20, pad=20)
        plt.xlabel('Theta/Beta Ratio (Mean)', fontsize=16)
        plt.ylabel('Participant', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        
        # Add border
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        
        # Save with tight layout
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'BigGraph_Horizontal_Stress_Comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Horizontal BigGraph saved to: {output_path}")
        print("✅ Tu gráfica horizontal ha sido generada correctamente.")
        
    except Exception as e:
        logger.error(f"Error generating BigGraph: {str(e)}")
    finally:
        plt.close('all')

def extract_user_id_from_filename(filename):
    """
    Extract user ID from result filename
    
    Args:
        filename: Filename to extract user ID from
    
    Returns:
        User ID extracted from filename
    """
    # Try different filename patterns
    if filename.startswith("User"):
        # Format like "User1_eeg_analysis_results.csv"
        return filename.split('_')[0]
    elif "results_" in filename:
        # Format like "eeg_analysis_results_User1.csv"
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.startswith("User"):
                return part
    
    # If no specific pattern matches, return filename without extension
    return os.path.splitext(filename)[0]

def process_existing_results(folder_path):
    """
    Process existing analysis results to create comprehensive comparison
    
    Args:
        folder_path: Path to folder containing result files
    
    Returns:
        Path to folder with generated comparison results
    """
    logger.info(f"Processing existing result files in: {folder_path}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(folder_path, "StressRatio", f"comprehensive_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Look for analysis result files
    all_files = os.listdir(folder_path)
    analysis_files = []
    
    # Try different patterns to find analysis files
    patterns = [
        "*eeg_analysis_results.csv",
        "*stress_analysis.csv",
        "*_results.csv"
    ]
    
    # First, try to look for stress_analysis files which have summary data
    stress_analysis_files = [f for f in all_files if "stress_analysis" in f.lower()]
    if stress_analysis_files:
        logger.info(f"Found {len(stress_analysis_files)} stress analysis files")
        analysis_files = stress_analysis_files
    else:
        # If no stress_analysis files, look for eeg_analysis_results
        eeg_results_files = [f for f in all_files if "eeg_analysis_results" in f.lower()]
        if eeg_results_files:
            logger.info(f"Found {len(eeg_results_files)} EEG analysis result files")
            analysis_files = eeg_results_files
        else:
            # Fallback to any CSV file that might contain results
            csv_files = [f for f in all_files if f.endswith(".csv")]
            logger.info(f"Found {len(csv_files)} CSV files (fallback)")
            analysis_files = csv_files
    
    if not analysis_files:
        logger.warning("No analysis files found")
        return None
    
    # Process each file and collect stress data
    all_users_data = []
    
    for file in analysis_files:
        file_path = os.path.join(folder_path, file)
        try:
            logger.info(f"Reading file: {file}")
            df = pd.read_csv(file_path)
            
            # Extract user ID from filename
            user_id = extract_user_id_from_filename(file)
            
            # Check if this is a stress_analysis file (single row) or a results file (multiple rows)
            if "stress_analysis" in file.lower() or len(df) == 1:
                # Single row with stress analysis summary
                row = df.iloc[0]
                if 'User' not in row:
                    row_dict = row.to_dict()
                    row_dict['User'] = user_id
                    all_users_data.append(row_dict)
                else:
                    all_users_data.append(row.to_dict())
            else:
                # Multiple rows with channel data, extract stress ratio
                if 'Stress_Ratio' in df.columns and 'Channel' in df.columns:
                    # Calculate mean stress ratio across all channels
                    mean_stress = df['Stress_Ratio'].mean()
                    std_stress = df['Stress_Ratio'].std()
                    min_stress = df['Stress_Ratio'].min()
                    max_stress = df['Stress_Ratio'].max()
                    
                    # Calculate frontal and parietal stress if possible
                    frontal_channels = df[df['Channel'].isin(['F3', 'Fz', 'F4'])]
                    parietal_channels = df[df['Channel'].isin(['P3', 'Pz', 'P4'])]
                    
                    frontal_mean = frontal_channels['Stress_Ratio'].mean() if not frontal_channels.empty else 0
                    parietal_mean = parietal_channels['Stress_Ratio'].mean() if not parietal_channels.empty else 0
                    
                    # Calculate threshold
                    suggested_threshold = mean_stress - (0.5 * std_stress)
                    
                    # Create user summary
                    user_summary = {
                        'User': user_id,
                        'mean': mean_stress,
                        'std': std_stress,
                        'min': min_stress,
                        'max': max_stress,
                        'frontal_mean': frontal_mean,
                        'parietal_mean': parietal_mean,
                        'suggested_threshold': suggested_threshold,
                        'overall_stress_level': "HIGH" if mean_stress < suggested_threshold else "NORMAL"
                    }
                    all_users_data.append(user_summary)
                else:
                    logger.warning(f"File {file} does not contain stress ratio data")
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")
    
    if not all_users_data:
        logger.warning("No user data could be extracted from files")
        return None
    
    # Create summary dataframe
    summary_df = pd.DataFrame(all_users_data)
    
    # Make sure required columns exist
    required_columns = ['User', 'mean', 'suggested_threshold']
    missing_columns = [col for col in required_columns if col not in summary_df.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        # Try to fix missing columns if possible
        if 'mean' not in summary_df.columns and 'Stress_Ratio' in summary_df.columns:
            summary_df['mean'] = summary_df['Stress_Ratio']
        if 'suggested_threshold' not in summary_df.columns and 'mean' in summary_df.columns and 'std' in summary_df.columns:
            summary_df['suggested_threshold'] = summary_df['mean'] - 0.5 * summary_df['std']
    
    # Check again if we have required columns
    missing_columns = [col for col in required_columns if col not in summary_df.columns]
    if missing_columns:
        logger.error(f"Still missing required columns after fixes: {missing_columns}")
        return None
    
    # Save summary to CSV
    summary_path = os.path.join(output_dir, "all_users_stress_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Comprehensive stress summary saved to: {summary_path}")
    
    # Generate the BigGraph
    generate_big_graph(summary_df, output_dir)
    
    return output_dir

def main():
    """
    Main function to find the Users_StressRatio folder and process data
    """
    print("\n===== EEG STRESS ANALYSIS COMPARISON TOOL =====")
    print("This tool will find the Users_StressRatio folder and analyze all user data")
    
    # Try to find the Users_StressRatio folder
    target_folder = "Users_StressRatio"
    folder_path = find_folder(target_folder)
    
    if folder_path:
        print(f"\nFound {target_folder} folder at: {folder_path}")
        
        # Process existing results in the folder
        output_dir = process_existing_results(folder_path)
        
        if output_dir:
            print(f"\nAnalysis complete! BigGraph and summary results saved to: {output_dir}")
            print("\nKey files generated:")
            print("1. BigGraph_User_Stress_Comparison.png - Vertical bar chart comparing all users")
            print("2. BigGraph_Horizontal_Stress_Comparison.png - Horizontal version for better visibility")
            print("3. all_users_stress_summary.csv - Comprehensive data table with all metrics")
        else:
            print("\nError: Could not process the data in the folder.")
    else:
        print(f"\nError: Could not find the {target_folder} folder.")
        
        # Try the alternate specific path as a fallback
        specific_path = "/Users/lawra/MINDLAND3/MINDMAX_PROJECT/Users_RawEEG"
        if os.path.exists(specific_path):
            print(f"Trying specific path: {specific_path}")
            output_dir = process_existing_results(specific_path)
            
            if output_dir:
                print(f"\nAnalysis complete! BigGraph and summary results saved to: {output_dir}")
            else:
                print("\nError: Could not process the data in the specific folder.")
        else:
            print("\nCould not find the target folder using any method.")
            
            # Ask user to enter path manually
            manual_path = input("\nPlease enter the path to Users_StressRatio folder manually: ").strip()
            if manual_path and os.path.exists(manual_path) and os.path.isdir(manual_path):
                output_dir = process_existing_results(manual_path)
                
                if output_dir:
                    print(f"\nAnalysis complete! BigGraph and summary results saved to: {output_dir}")
                else:
                    print("\nError: Could not process the data in the manually specified folder.")
            else:
                print("\nInvalid path or directory does not exist.")

if __name__ == "__main__":
    # Configure console logger for user-friendly output
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    
    # Add console handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console)
    
    main()