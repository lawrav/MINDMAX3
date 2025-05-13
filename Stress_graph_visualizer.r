import matplotlib.pyplot as plt
import seaborn as sns

def plot_eeg_metrics(df):
    plt.figure(figsize=(12, 6))
    
    # Plot Beta and Theta Relative Power
    plt.subplot(2, 1, 1)
    sns.lineplot(data=df, x=df.index, y='beta_relative', label='Beta Relative Power')
    sns.lineplot(data=df, x=df.index, y='theta_relative', label='Theta Relative Power')
    plt.title('Relative Power of Beta and Theta Waves')
    plt.ylabel('Power')
    plt.xlabel('Time/Samples')
    
    # Plot Beta-to-Theta Ratio (Stress Indicator)
    plt.subplot(2, 1, 2)
    sns.lineplot(data=df, x=df.index, y='btr', color='red', label='Beta/Theta Ratio (BTR)')
    plt.axhline(y=1.5, color='orange', linestyle='--', label='Moderate Stress Threshold')
    plt.axhline(y=3.0, color='red', linestyle='--', label='High Stress Threshold')
    plt.title('Beta-to-Theta Ratio (Stress Level)')
    plt.ylabel('BTR')
    plt.xlabel('Time/Samples')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_eeg_metrics(processed_data)
