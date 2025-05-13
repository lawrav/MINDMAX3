import socket
import threading
import time
import numpy as np
import csv
from pathlib import Path
from pylsl import StreamInlet, resolve_byprop

# --- Configuración ---
HOST = "127.0.0.1"
PORT = 6666
ELECTRODOS = ["F3", "Fz", "F4", "C3", "C4", "P3", "Pz", "P4"]

SCALE_FACTOR_EEG = (4500000) / 24 / (2**23 - 1)

data_folder = "./datos_guardados"
Path(data_folder).mkdir(parents=True, exist_ok=True)

ciclo_actual = 0
calibration_data = {e: [] for e in ELECTRODOS}
baseline = None
electrodo_baseline = None
inlet = None
muestreo_activo = False
nivel_actual = 3
tiempo_inicio_juego = None
datos_juego = []
datos_raw = []
segundos_procesados = set()
tiempo_restante = 900
modo = "CALIBRACIÓN"
temporizador_iniciado = False
nombre_participante = "anonimo"

def calcular_ce(theta, alpha, beta):
    return [b / (t + a) if (t + a) != 0 else 0 for b, t, a in zip(beta, theta, alpha)]

def leer_aura():
    global inlet
    if inlet is None:
        return None, None
    muestra, timestamp = inlet.pull_sample(timeout=0.5)
    if muestra and len(muestra) >= 40:
        datos_escalados = [float(x) * SCALE_FACTOR_EEG for x in muestra[:40]]
        theta = muestra[10:18]
        alpha = muestra[18:26]
        beta = muestra[26:34]
        ce = calcular_ce(theta, alpha, beta)
        return ce, datos_escalados
    return None, None

def verificar_conexion():
    global inlet
    print("[INFO] Buscando flujo AURA_Power...")
    streams = resolve_byprop('name', 'AURA_Power')
    if not streams:
        return False
    inlet_temp = StreamInlet(streams[0])
    muestra, _ = inlet_temp.pull_sample(timeout=3)
    if muestra and len(muestra) >= 40:
        inlet = inlet_temp
        print("[OK] Conectado al flujo EEG.")
        return True
    return False

def enviar_senal_a_unity(valor):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((HOST, 5555))
        client.sendall(str(valor).encode())
        client.close()
        print(f"[Unity] Señal enviada: {valor}")
    except Exception as e:
        print(f"[ERROR] Fallo al enviar a Unity: {e}")

def muestreo_periodico():
    global tiempo_inicio_juego, segundos_procesados, muestreo_activo, nivel_actual
    segundos_procesados.clear()
    segundo_actual = -1
    print("[INFO] Iniciando muestreo de juego...")
    while muestreo_activo:
        tiempo_transcurrido = int(time.time() - tiempo_inicio_juego)
        if tiempo_transcurrido != segundo_actual:
            segundo_actual = tiempo_transcurrido
            ce, raw = leer_aura()
            if ce and electrodo_baseline in ELECTRODOS:
                idx = ELECTRODOS.index(electrodo_baseline)
                ce_valor = ce[idx]
                datos_juego.append((tiempo_transcurrido, ce_valor))
                print(f"[DATA] Segundo {tiempo_transcurrido}: CE = {ce_valor:.3f}")
                if raw:
                    datos_raw.append((tiempo_transcurrido, raw))
                if tiempo_transcurrido % 10 == 0 and tiempo_transcurrido not in segundos_procesados:
                    segundos_procesados.add(tiempo_transcurrido)
                    if ce_valor > baseline * 1.05 and nivel_actual < 5:
                        nivel_actual += 1
                        enviar_senal_a_unity(2)
                    elif ce_valor < baseline * 0.95 and nivel_actual > 1:
                        nivel_actual -= 1
                        enviar_senal_a_unity(0)
                    else:
                        enviar_senal_a_unity(1)
        time.sleep(0.1)

def servidor_python():
    global ciclo_actual, baseline, electrodo_baseline, muestreo_activo, tiempo_inicio_juego, modo, temporizador_iniciado
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.bind((HOST, PORT))
    servidor.listen(5)
    print(f"[SERVIDOR] Escuchando en {HOST}:{PORT}...")
    while True:
        conn, addr = servidor.accept()
        with conn:
            data = conn.recv(1024).decode("utf-8").strip()
            print(f"[SEÑAL RECIBIDA] {data}")
            if data == "a" and ciclo_actual < 10:
                if not temporizador_iniciado:
                    tiempo_inicio_juego = time.time()
                    temporizador_iniciado = True
                ce, _ = leer_aura()
                if ce:
                    for i, e in enumerate(ELECTRODOS):
                        calibration_data[e].append(ce[i])
                    ciclo_actual += 1
                    print(f"[CALIBRACIÓN] Ciclo {ciclo_actual}/10")
                if ciclo_actual == 10:
                    promedios = {e: np.mean(val) for e, val in calibration_data.items() if val}
                    electrodo_baseline = max(promedios, key=promedios.get)
                    baseline = promedios[electrodo_baseline]
                    modo = "JUEGO"
                    print(f"[BASELINE] Electrodo: {electrodo_baseline}, CE promedio: {baseline:.3f}")
            elif data == "i":
                if not muestreo_activo:
                    muestreo_activo = True
                    threading.Thread(target=muestreo_periodico, daemon=True).start()
            elif data == "t":
                muestreo_activo = False
                print("[INFO] Muestreo detenido.")
            elif data == "s":
                guardar_datos()

def guardar_datos():
    global datos_juego, datos_raw
    if not nombre_participante or not data_folder:
        print("[ERROR] Nombre o carpeta no definidos.")
        return

    ruta = Path(data_folder) / f"{nombre_participante}_AdaptiveGame.csv"
    with open(ruta, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"Electrodo baseline: {electrodo_baseline}"])
        writer.writerow([f"CE promedio: {baseline:.3f}"])
        writer.writerow([])
        writer.writerow(["Segundo", "CE"])
        writer.writerows(datos_juego)
    print(f"[OK] Datos CE guardados en {ruta}")

    ruta_raw = Path(data_folder) / f"{nombre_participante}_AdaptiveGame_raw.csv"
    with open(ruta_raw, "w", newline="") as f:
        writer = csv.writer(f)
        encabezados = ["Segundo"]
        for i in range(8):
            for banda in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
                encabezados.append(f"Canal{i+1}_{banda}")
        writer.writerow(encabezados)
        for segundo, raw_vals in datos_raw:
            writer.writerow([segundo] + raw_vals)
    print(f"[OK] Datos RAW guardados en {ruta_raw}")

# --- Inicio del programa ---
if __name__ == "__main__":
    print("==== ADAPTIVE GAME CONSOLE MODE ====")
    nombre_participante = input("Ingresa el nombre del participante: ").strip()
    if verificar_conexion():
        threading.Thread(target=servidor_python, daemon=True).start()
        print("[LISTO] Esperando señales desde Unity...")
        while True:
            time.sleep(1)
    else:
        print("[ERROR] No se pudo iniciar porque no se detectó el EEG.")
