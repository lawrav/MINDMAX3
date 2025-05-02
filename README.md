# 🧠 EEG Stress Analysis Toolkit

This Python script processes EEG data from CSV files to analyze stress indicators using frequency band analysis. It is optimized for signals from electrodes such as F3, Fz, F4, C3, C4, P3, Pz, and P4.

## 🔍 Key Features

* **Preprocessing & Band Analysis**:

  * Computes Power Spectral Density (PSD) using the Welch method.
  * Extracts absolute and relative power across Delta, Theta, Alpha, Beta, and Gamma bands.
  * Calculates Theta/Beta stress ratio and Frontal Alpha Asymmetry for emotional insights.

* **Stress Evaluation**:

  * Determines overall stress level based on Theta/Beta ratios.
  * Differentiates stress signals from frontal vs. parietal brain regions.

* **Visualizations**:

  * EEG raw signal plots.
  * PSD plots per channel with frequency band highlights.
  * Stress ratio bar graphs with threshold markers.
  * EEG metric heatmaps for comparative analysis.

* **Exportable Outputs**:

  * Processed results (`eeg_analysis_results.csv`)
  * Cleaned raw EEG data (`raw_eeg_data.csv`)
  * Stress summary (`stress_analysis.csv`)
  * Graphs saved as PNGs

## 🧪 How to Use

Run the script and provide the path to your EEG `.csv` file when prompted:

```bash
python TestProcessDataOK.py
```

If no file path is given, it uses a default sample. Results are stored in a timestamped folder.

## 🧰 Dependencies

* Python ≥ 3.7
* `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

Install via:

```bash
pip install pandas numpy matplotlib seaborn scipy
```

---

Developed for use with EEG signal processing and mental health applications.
