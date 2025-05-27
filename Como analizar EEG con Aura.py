import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import simpson as simps
import os
from datetime import datetime
import seaborn as sns

def process_eeg_data(csv_file, output_dir="output"):
    """
    Procesa datos de EEG para extraer bandas de frecuencia y métricas de estrés
    
    Args:
        csv_file: Ruta al archivo CSV con datos EEG
        output_dir: Directorio donde guardar los resultados
    
    Returns:
        Tuple con DataFrame de resultados y DataFrame de datos raw transformados
    """
    print(f"Procesando archivo: {csv_file}")
    
    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Cargar los datos, saltando la fila de unidades
    df = pd.read_csv(csv_file, skiprows=1, quotechar="'")
    
    # Parámetros
    fs = 250  # Frecuencia de muestreo (Hz) - ajustar si tu dispositivo usa otra
    window_size = 4 * fs  # Ventanas de 4 segundos
    
    # Bandas de frecuencia
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45)
    }
    
    results = []
    raw_data_dict = {}
    psd_data_dict = {}
    
    # Procesar cada canal
    channels = ['F3', 'Fz', 'F4', 'C3', 'P3', 'C4', 'Pz', 'P4']
    channel_map = {
        'F3': 'uV',
        'Fz': 'uV.1',
        'F4': 'uV.2',
        'C3': 'uV.3',
        'P3': 'uV.4',
        'C4': 'uV.5',
        'Pz': 'uV.6',
        'P4': 'uV.7'
    }
    
    # Guardar los datos raw para graficar
    for channel in channels:
        col_name = channel_map[channel]
        if col_name in df.columns:
            raw_data_dict[channel] = df[col_name].astype(float).values
        else:
            print(f"Advertencia: columna {col_name} no encontrada para el canal {channel}")
            print(f"Columnas disponibles: {df.columns.tolist()}")
            continue
    
    for channel in channels:
        # Obtener la señal usando el nombre de columna mapeado
        if channel_map[channel] not in df.columns:
            continue
            
        signal = df[channel_map[channel]].values
        
        # Convertir valores de string a float si es necesario
        signal = signal.astype(float)
        
        # Calcular PSD usando el método de Welch
        freqs, psd = welch(signal, fs=fs, nperseg=window_size)
        
        # Guardar los datos de PSD para graficar
        psd_data_dict[channel] = {'freqs': freqs, 'psd': psd}
        
        # Calcular potencia absoluta en cada banda
        band_powers = {}
        total_power = simps(psd, freqs)
        
        for band, (low, high) in bands.items():
            # Encontrar índices de frecuencias en la banda
            idx = np.logical_and(freqs >= low, freqs <= high)
            
            # Calcular potencia absoluta integrando el PSD
            band_power = simps(psd[idx], freqs[idx])
            band_powers[band] = band_power
        
        # Calcular potencias relativas (métricas de estrés)
        beta_relative = band_powers['Beta'] / total_power if total_power > 0 else 0
        theta_relative = band_powers['Theta'] / total_power if total_power > 0 else 0
        alpha_relative = band_powers['Alpha'] / total_power if total_power > 0 else 0
        
        # Calcular ratio de estrés (métrica común)
        # Usamos theta/beta como indicador inverso de estrés (valores más altos = menos estrés)
        stress_ratio = theta_relative / beta_relative if beta_relative > 0 else 0
        
        # Calcular ratio frontal alpha asimétrico (indicador de valencias emocionales)
        # Solo aplica para canales F3 y F4
        frontal_alpha_asymmetry = 0
        if channel == 'F3':
            f3_alpha = band_powers['Alpha']
            if 'F4' in channel_map and channel_map['F4'] in df.columns:
                # Guardamos para usar cuando procesemos F4
                raw_data_dict['F3_alpha'] = f3_alpha
        elif channel == 'F4':
            f4_alpha = band_powers['Alpha']
            if 'F3_alpha' in raw_data_dict:
                # Solo calculamos si ya procesamos F3
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
    
    results_df = pd.DataFrame(results)
    
    # Convertir datos raw a DataFrame para análisis y gráficos
    max_length = max(len(val) for val in raw_data_dict.values() if isinstance(val, np.ndarray))
    raw_data_processed = {}
    
    for channel, data in raw_data_dict.items():
        if isinstance(data, np.ndarray):
            # Solo procesar arrays numpy (datos de canales)
            if len(data) < max_length:
                # Rellenar con NaN si es necesario
                padded_data = np.pad(data, (0, max_length - len(data)), 
                                     mode='constant', constant_values=np.nan)
                raw_data_processed[channel] = padded_data
            else:
                raw_data_processed[channel] = data
    
    raw_df = pd.DataFrame(raw_data_processed)
    
    return results_df, raw_df, psd_data_dict

def analyze_stress_metrics(results_df):
    """
    Analiza las métricas de estrés y propone umbrales
    
    Args:
        results_df: DataFrame con los resultados procesados
    
    Returns:
        Dict con los análisis y umbrales recomendados
    """
    stress_analysis = {}
    
    # Extraer métricas de estrés para todos los canales
    stress_ratios = results_df['Stress_Ratio'].values
    
    # Estadísticas básicas
    stress_min = np.min(stress_ratios)
    stress_max = np.max(stress_ratios)
    stress_mean = np.mean(stress_ratios)
    stress_median = np.median(stress_ratios)
    stress_std = np.std(stress_ratios)
    
    # Analizar por grupos de canales (frontales vs parietales)
    frontal_mask = results_df['Channel'].isin(['F3', 'Fz', 'F4'])
    parietal_mask = results_df['Channel'].isin(['P3', 'Pz', 'P4'])
    
    frontal_stress = results_df.loc[frontal_mask, 'Stress_Ratio'].mean()
    parietal_stress = results_df.loc[parietal_mask, 'Stress_Ratio'].mean()
    
    # Proponer umbral basado en estadísticas
    # Típicamente, valores menores indican mayor estrés para el ratio theta/beta
    suggested_threshold = stress_mean - (0.5 * stress_std)
    
    # Evaluar nivel de estrés general
    overall_stress_level = "ALTO" if stress_mean < suggested_threshold else "NORMAL"
    
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
    
    return stress_analysis

def generate_raw_plots(raw_df, output_dir):
    """
    Genera gráficos para los datos crudos de EEG
    
    Args:
        raw_df: DataFrame con datos crudos de EEG
        output_dir: Directorio donde guardar los gráficos
    """
    plt.figure(figsize=(15, 10))
    
    # Crear escala de tiempo para el eje X (en segundos)
    fs = 250  # Frecuencia de muestreo
    time_sec = np.arange(len(raw_df)) / fs
    
    # Graficar solo los primeros 10 segundos para mejor visualización
    plot_samples = min(10 * fs, len(raw_df))
    
    # Definir los canales a graficar
    channels = raw_df.columns.tolist()
    
    # Crear subplots para cada canal
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, n_channels*2), sharex=True)
    
    # Si solo hay un canal, axes no será un array
    if n_channels == 1:
        axes = [axes]
    
    for i, channel in enumerate(channels):
        axes[i].plot(time_sec[:plot_samples], raw_df[channel].values[:plot_samples])
        axes[i].set_title(f'Canal {channel}')
        axes[i].set_ylabel('Amplitud (uV)')
        
    axes[-1].set_xlabel('Tiempo (s)')
    plt.tight_layout()
    
    # Guardar la figura
    plt.savefig(os.path.join(output_dir, 'raw_eeg_data.png'), dpi=300)
    plt.close()

def generate_psd_plots(psd_data_dict, bands, output_dir):
    """
    Genera gráficos de densidad espectral de potencia (PSD)
    
    Args:
        psd_data_dict: Diccionario con datos PSD por canal
        bands: Diccionario de bandas de frecuencia
        output_dir: Directorio donde guardar los gráficos
    """
    # Crear figura para todos los canales
    n_channels = len(psd_data_dict)
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, n_channels*2), sharex=True)
    
    # Si solo hay un canal, axes no será un array
    if n_channels == 1:
        axes = [axes]
    
    # Colores para las bandas
    band_colors = {
        'Delta': 'blue',
        'Theta': 'green',
        'Alpha': 'red',
        'Beta': 'purple',
        'Gamma': 'orange'
    }
    
    i = 0
    for channel, data in psd_data_dict.items():
        freqs = data['freqs']
        psd = data['psd']
        
        # Graficar el PSD
        axes[i].semilogy(freqs, psd)
        axes[i].set_title(f'PSD del Canal {channel}')
        axes[i].set_ylabel('PSD (uV²/Hz)')
        
        # Colorear las diferentes bandas de frecuencia
        ylim = axes[i].get_ylim()
        for band, (low, high) in bands.items():
            axes[i].fill_between(
                freqs[(freqs >= low) & (freqs <= high)],
                psd[(freqs >= low) & (freqs <= high)],
                color=band_colors[band],
                alpha=0.3,
                label=band
            )
        axes[i].set_ylim(ylim)
        
        # Agregar leyenda
        if i == 0:
            axes[i].legend()
        
        i += 1
        
    axes[-1].set_xlabel('Frecuencia (Hz)')
    plt.tight_layout()
    
    # Guardar la figura
    plt.savefig(os.path.join(output_dir, 'psd_by_channel.png'), dpi=300)
    plt.close()

def generate_stress_plots(results_df, stress_analysis, output_dir):
    """
    Genera gráficos de métricas de estrés
    
    Args:
        results_df: DataFrame con resultados de análisis
        stress_analysis: Dict con análisis de estrés
        output_dir: Directorio donde guardar los gráficos
    """
    # Gráfico de barras del ratio de estrés por canal
    plt.figure(figsize=(12, 6))
    
    # Crear un color especial para los canales bajo el umbral (más estresados)
    colors = ['red' if val < stress_analysis['suggested_threshold'] else 'blue' 
              for val in results_df['Stress_Ratio']]
    
    sns.barplot(x='Channel', y='Stress_Ratio', data=results_df, palette=colors)
    
    # Dibujar línea de umbral
    plt.axhline(y=stress_analysis['suggested_threshold'], color='r', linestyle='--', 
                label=f'Umbral de Estrés: {stress_analysis["suggested_threshold"]:.4f}')
    
    plt.title('Ratio de Estrés por Canal (Theta/Beta)\nValores más bajos indican mayor estrés')
    plt.ylabel('Ratio Theta/Beta')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # Agregar anotaciones con los valores
    for i, v in enumerate(results_df['Stress_Ratio']):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Guardar la figura
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stress_ratio_by_channel.png'), dpi=300)
    plt.close()
    
    # Crear un heatmap de todas las métricas
    plt.figure(figsize=(14, 8))
    metric_cols = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 
                  'Beta_Relative', 'Theta_Relative', 'Alpha_Relative', 'Stress_Ratio']
    
    # Crear un DataFrame pivoteado para el heatmap
    heatmap_df = results_df.pivot(index='Channel', columns=None, values=metric_cols)
    
    # Estandarizar los valores para mejor visualización
    normalized_df = (heatmap_df - heatmap_df.mean()) / heatmap_df.std()
    
    sns.heatmap(normalized_df, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title('Mapa de Calor de Métricas EEG por Canal (Valores Normalizados)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eeg_metrics_heatmap.png'), dpi=300)
    plt.close()

def main(input_file):
    """
    Función principal para procesar y analizar datos EEG
    
    Args:
        input_file: Ruta al archivo CSV con datos EEG
    """
    # Crear directorio con timestamp para los resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"eeg_analysis_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Bandas de frecuencia
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45)
    }
    
    # Procesar los datos
    print("Procesando datos EEG...")
    results_df, raw_df, psd_data_dict = process_eeg_data(input_file, output_dir)
    
    # Guardar resultados procesados
    results_path = os.path.join(output_dir, "eeg_analysis_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Resultados guardados en: {results_path}")
    
    # Guardar datos raw transformados
    raw_path = os.path.join(output_dir, "raw_eeg_data.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Datos raw guardados en: {raw_path}")
    
    # Analizar las métricas de estrés
    print("Analizando métricas de estrés...")
    stress_analysis = analyze_stress_metrics(results_df)
    
    # Generar gráficos
    print("Generando gráficos...")
    generate_raw_plots(raw_df, output_dir)
    generate_psd_plots(psd_data_dict, bands, output_dir)
    generate_stress_plots(results_df, stress_analysis, output_dir)
    
    # Guardar análisis de estrés
    stress_df = pd.DataFrame([stress_analysis])
    stress_path = os.path.join(output_dir, "stress_analysis.csv")
    stress_df.to_csv(stress_path, index=False)
    print(f"Análisis de estrés guardado en: {stress_path}")
    
    # Mostrar análisis de estrés
    print("\n=== ANÁLISIS DE ESTRÉS ===")
    print(f"Valor mínimo: {stress_analysis['min']:.4f}")
    print(f"Valor máximo: {stress_analysis['max']:.4f}")
    print(f"Valor promedio: {stress_analysis['mean']:.4f}")
    print(f"Valor mediana: {stress_analysis['median']:.4f}")
    print(f"Desviación estándar: {stress_analysis['std']:.4f}")
    print(f"Promedio canales frontales: {stress_analysis['frontal_mean']:.4f}")
    print(f"Promedio canales parietales: {stress_analysis['parietal_mean']:.4f}")
    print(f"Umbral de estrés recomendado: {stress_analysis['suggested_threshold']:.4f}")
    print(f"Nivel de estrés general: {stress_analysis['overall_stress_level']}")
    
    if stress_analysis['overall_stress_level'] == "ALTO":
        print("\n⚠️ ALERTA: ¡NIVEL DE ESTRÉS ELEVADO DETECTADO! ⚠️")
    
    print(f"\nTodos los resultados del análisis se guardaron en el directorio: {output_dir}")
    return output_dir

# Para probar el código, ejecuta la siguiente línea con la ruta a tu archivo CSV
if __name__ == "__main__":
    # Ruta al archivo CSV con datos EEG
    # Reemplaza con la ruta de tu archivo
    input_file = input("Introduce la ruta completa al archivo CSV de EEG: ")
    
    # Si el usuario no proporciona una ruta, usar un valor por defecto
    if not input_file.strip():
        print("Usando ruta de ejemplo...")
        input_file = 'AURA_RAW___2025-03-10___16;04;11.csv'
    
    # Comprobar si el archivo existe
    if not os.path.isfile(input_file):
        print(f"Error: El archivo {input_file} no existe.")
        print("Por favor, proporciona una ruta válida al archivo CSV.")
        exit(1)
    
    # Procesar el archivo
    main(input_file)