import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def calcular_snr(audio_senal, audio_ruido):
    potencia_senal = np.mean(audio_senal ** 2)
    potencia_ruido = np.mean(audio_ruido ** 2)

    if potencia_ruido == 0:
        return float("inf")

    return 10 * np.log10(potencia_senal / potencia_ruido)


def mostrar_espectrograma(audio, frecuencia_muestreo, titulo):
    frecuencias, tiempos, potencia = spectrogram(
        audio,
        fs=frecuencia_muestreo
    )

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(tiempos, frecuencias, 10 * np.log10(potencia + 1e-12))
    plt.title(titulo)
    plt.ylabel("Frecuencia (Hz)")
    plt.xlabel("Tiempo (s)")
    plt.ylim(0, 4000)
    plt.colorbar(label="Potencia (dB)")
    plt.tight_layout()
    plt.show()

def evaluar_snr(ref_entrada, audio_entrada,
                ref_salida, audio_salida, frecuencia_muestreo):

    inicio = int(0.8 * frecuencia_muestreo)
    fin = int(2.2 * frecuencia_muestreo)

    voz_entrada = ref_entrada[inicio:fin]
    ruido_entrada = audio_entrada[inicio:fin] - voz_entrada

    voz_salida = ref_salida[inicio:fin]
    ruido_salida = audio_salida[inicio:fin] - voz_salida

    snr_entrada = calcular_snr(voz_entrada, ruido_entrada)
    snr_salida = calcular_snr(voz_salida, ruido_salida)

    print("\n--- Evaluación SNR ---")
    print("SNR de entrada:", round(snr_entrada, 2), "dB")
    print("SNR de salida:", round(snr_salida, 2), "dB")
    print("Mejora:", round(snr_salida - snr_entrada, 2), "dB")