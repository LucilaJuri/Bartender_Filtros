
import numpy as np
import matplotlib.pyplot as plt
from filters import crear_filtro_pasabanda, aplicar_filtro
from vad import DetectorVoz
from noise_reduction import reducir_ruido
from agc import control_ganancia
import soundfile as sf
from filters import aplicar_preenfasis
from analysis import mostrar_espectrograma
from analysis import mostrar_espectrograma, evaluar_snr

# ==========================================
# 1. CONFIGURACION DEL AUDIO
# ==========================================

frecuencia_muestreo = 16000  # Hz
duracion = 3                  # segundos

# Generar eje temporal
tiempo = np.arange(0, duracion, 1/frecuencia_muestreo)

# Crear señal inicial en silencio
audio = np.zeros(len(tiempo))

# ==========================================
# 2. GENERAR UNA VOZ SIMULADA
# ==========================================

# La voz comienza en 0.8 segundos
# y termina en 2.2 segundos

inicio_voz = 0.8
fin_voz = 2.2

# Seleccionar las muestras correspondientes a la voz
indices_voz = (tiempo >= inicio_voz) & (tiempo <= fin_voz)

# Generar una señal con varias frecuencias
t_voz = tiempo[indices_voz] - inicio_voz

voz = np.zeros(len(t_voz))

# Tres segmentos de voz simulada
segmentos = [
    (0.0, 0.35, 180),
    (0.55, 0.95, 220),
    (1.15, 1.40, 200),
    (2.0, 2.35, 190)
]

for inicio_seg, fin_seg, frecuencia in segmentos:

    indices = (t_voz >= inicio_seg) & (t_voz < fin_seg)

    t_segmento = t_voz[indices] - inicio_seg

    envolvente = np.sin(
        np.pi * np.arange(len(t_segmento)) / len(t_segmento)
    )

    voz[indices] = envolvente * (
        0.6 * np.sin(2 * np.pi * frecuencia * t_segmento) +
        0.25 * np.sin(2 * np.pi * 800 * t_segmento) +
        0.15 * np.sin(2 * np.pi * 1500 * t_segmento)
    )

# Guardar la voz limpia como referencia
audio_referencia = audio.copy()
audio_referencia[indices_voz] = voz

# Generar ruido
ruido = np.random.normal(0, 0.03, len(audio))

# Audio original con ruido
audio = audio_referencia + ruido


# ==========================================
# 4. PROCESAMIENTO POR BLOQUES DE 20 ms
# ==========================================

duracion_bloque = 0.020

muestras_por_bloque = int(
    frecuencia_muestreo * duracion_bloque
)

filtro = crear_filtro_pasabanda(frecuencia_muestreo)

# Estado inicial del filtro
estado = np.zeros((filtro.shape[0], 2))

# Almacenará toda la señal filtrada
audio_filtrado = np.zeros(len(audio))

# Procesar bloque por bloque
for inicio in range(0, len(audio), muestras_por_bloque):

    fin = min(inicio + muestras_por_bloque, len(audio))

    bloque = audio[inicio:fin]

    # Aplicar filtro conservando el estado anterior
    bloque_filtrado, estado = aplicar_filtro(
        bloque,
        filtro,
        estado
    )

    audio_filtrado[inicio:fin] = bloque_filtrado

print("Procesamiento por bloques finalizado")
print("Duración de cada bloque:", duracion_bloque * 1000, "ms")
print("Muestras por bloque:", muestras_por_bloque)
print("Cantidad de bloques:", int(np.ceil(len(audio) / muestras_por_bloque)))
# Filtrar también la referencia limpia
estado_ref = np.zeros((filtro.shape[0], 2))
audio_referencia_filtrada = np.zeros(len(audio_referencia))

for inicio in range(0, len(audio_referencia), muestras_por_bloque):
    fin = min(inicio + muestras_por_bloque, len(audio_referencia))

    bloque = audio_referencia[inicio:fin]

    bloque_filtrado, estado_ref = aplicar_filtro(
        bloque, filtro, estado_ref
    )

    audio_referencia_filtrada[inicio:fin] = bloque_filtrado

# Reducir ruido del audio
audio_limpio = reducir_ruido(audio_filtrado, frecuencia_muestreo)

print("Reducción de ruido finalizada")

audio_preenfatizado = aplicar_preenfasis(audio_limpio)

# Aplicar control automático de ganancia
audio_agc = control_ganancia(audio_limpio, frecuencia_muestreo)
print("Control automático de ganancia finalizado")

evaluar_snr(
    audio_referencia,
    audio,
    reducir_ruido(audio_referencia_filtrada, frecuencia_muestreo),
    audio_limpio,
    frecuencia_muestreo
)

# Guardar audio procesado
sf.write("resultados/audio_agc.wav", audio_agc, frecuencia_muestreo)

mostrar_espectrograma(
    audio,
    frecuencia_muestreo,
    "Espectrograma del audio original"
)

mostrar_espectrograma(
    audio_agc,
    frecuencia_muestreo,
    "Espectrograma del audio procesado"
)

import soundfile as sf

sf.write("resultados/audio_limpio.wav", audio_limpio, frecuencia_muestreo)
sf.write("resultados/audio_original.wav", audio, frecuencia_muestreo)

print("Audio limpio guardado correctamente")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(audio)
plt.title("Audio original")
plt.ylabel("Amplitud")

plt.subplot(2, 1, 2)
plt.plot(audio_limpio)
plt.title("Audio después de reducción de ruido")
plt.ylabel("Amplitud")
plt.xlabel("Muestras")

plt.tight_layout()
plt.show()



# ==========================================
# 5. DETECCION DE ACTIVIDAD DE VOZ
# ==========================================

detector = DetectorVoz(
    umbral_inicio=0.025,
    umbral_fin=0.045,
    bloques_silencio=15,
    bloques_minimos=5
)

bloques_inicio = []
bloques_fin = []

for inicio in range(0, len(audio_filtrado), muestras_por_bloque):

    fin = min(inicio + muestras_por_bloque, len(audio_filtrado))

    bloque = audio_filtrado[inicio:fin]

    rms, inicio_detectado, fin_detectado = detector.procesar_bloque(bloque)
    if 1.9 < inicio / frecuencia_muestreo < 2.6:
        print("RMS segundo segmento:", round(rms, 4))

    if inicio >= int(2.2 * frecuencia_muestreo):
        if inicio % (muestras_por_bloque * 5) == 0:
            print("RMS después de la voz:", round(rms, 4))

    if inicio_detectado:
        tiempo_inicio = inicio / frecuencia_muestreo
        bloques_inicio.append(tiempo_inicio)
        print("Inicio de voz:", round(tiempo_inicio, 3), "segundos")

    if fin_detectado:
        tiempo_fin = fin / frecuencia_muestreo
        bloques_fin.append(tiempo_fin)
        print("Fin de voz:", round(tiempo_fin, 3), "segundos")

# ==========================================
# 6. COMPARAR SEÑAL ORIGINAL Y FILTRADA
# ==========================================

plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(tiempo, audio)
plt.title("Señal original con ruido")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(tiempo, audio_filtrado)
plt.title("Señal después del filtro pasa banda")
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Amplitud")
plt.grid()

plt.tight_layout()
plt.show()
