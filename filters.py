
from scipy.signal import butter, sosfilt

import numpy as np


def crear_filtro_pasabanda(frecuencia_muestreo):
    """
    Diseña un filtro pasa banda Butterworth.
    Conserva las frecuencias entre 300 y 3400 Hz.
    """

    frecuencia_baja = 300
    frecuencia_alta = 3400
    orden = 4

    filtro = butter(
        orden,
        [frecuencia_baja, frecuencia_alta],
        btype="bandpass",
        fs=frecuencia_muestreo,
        output="sos"
    )

    return filtro


def aplicar_filtro(audio, filtro, estado=None):
    """
    Aplica el filtro a una señal de audio.
    Permite conservar el estado interno entre bloques.
    """

    if estado is None:
        audio_filtrado = sosfilt(filtro, audio)
        return audio_filtrado

    audio_filtrado, nuevo_estado = sosfilt(
        filtro,
        audio,
        zi=estado
    )

    return audio_filtrado, nuevo_estado

def aplicar_preenfasis(audio, alpha=0.97):
    audio_preenfatizado = np.append(
        audio[0],
        audio[1:] - alpha * audio[:-1]
    )
    return audio_preenfatizado