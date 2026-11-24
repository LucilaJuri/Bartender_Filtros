
import numpy as np


def control_ganancia(audio, frecuencia_muestreo):

    duracion_bloque = 0.020
    muestras_por_bloque = int(frecuencia_muestreo * duracion_bloque)

    nivel_objetivo = 0.08
    ganancia_maxima = 2.0
    ganancia = 1.0
    suavizado = 0.95

    audio_salida = np.zeros_like(audio)

    for inicio in range(0, len(audio), muestras_por_bloque):

        fin = min(inicio + muestras_por_bloque, len(audio))
        bloque = audio[inicio:fin]

        rms = np.sqrt(np.mean(bloque**2))

        if rms > 0.001:
            ganancia_deseada = nivel_objetivo / rms
            ganancia_deseada = min(ganancia_deseada, ganancia_maxima)
        else:
            ganancia_deseada = 1.0

        ganancia = suavizado * ganancia + (1 - suavizado) * ganancia_deseada

        audio_salida[inicio:fin] = bloque * ganancia

    audio_salida = np.clip(audio_salida, -1.0, 1.0)

    return audio_salida
