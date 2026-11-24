
import numpy as np
from scipy.signal import stft, istft


def reducir_ruido(audio, frecuencia_muestreo):

    # Transformar el audio al dominio de la frecuencia
    frecuencias, tiempos, espectro = stft(
        audio,
        fs=frecuencia_muestreo,
        nperseg=512,
        noverlap=256
    )

    # Identificar los primeros 0.5 segundos como ruido
    muestras_ruido = int(0.5 * frecuencia_muestreo)

    # Calcular la STFT del ruido
    _, _, espectro_ruido = stft(
        audio[:muestras_ruido],
        fs=frecuencia_muestreo,
        nperseg=512,
        noverlap=256
    )

    # Estimar el nivel promedio del ruido
    perfil_ruido = np.mean(np.abs(espectro_ruido), axis=1)

    # Obtener la amplitud del espectro original
    amplitud = np.abs(espectro)

    # Restar el perfil de ruido
    amplitud_limpia = np.maximum(
        amplitud - perfil_ruido[:, np.newaxis], 0
    )

    # Conservar la fase original
    fase = np.angle(espectro)

    # Reconstruir el espectro complejo
    espectro_limpio = amplitud_limpia * np.exp(1j * fase)

    # Reconstruir el audio en el dominio del tiempo
    _, audio_limpio = istft(
        espectro_limpio,
        fs=frecuencia_muestreo,
        nperseg=512,
        noverlap=256
    )

    # Ajustar la longitud al audio original
    audio_limpio = audio_limpio[:len(audio)]

    return audio_limpio

