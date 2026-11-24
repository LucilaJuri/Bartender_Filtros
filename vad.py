
import numpy as np


class DetectorVoz:

    def __init__(
        self,
        umbral_inicio=0.025,
        umbral_fin=0.045,
        bloques_silencio=20,
        bloques_minimos=5
    ):

        self.umbral_inicio = umbral_inicio
        self.umbral_fin = umbral_fin

        self.bloques_silencio = bloques_silencio
        self.bloques_minimos = bloques_minimos

        self.hablando = False
        self.contador_silencio = 0
        self.contador_voz = 0

    def procesar_bloque(self, bloque):

        # Calcular energía RMS del bloque
        rms = np.sqrt(np.mean(bloque ** 2))

        inicio_detectado = False
        fin_detectado = False

        if not self.hablando:

            if rms >= self.umbral_inicio:

                self.contador_voz += 1

                if self.contador_voz >= self.bloques_minimos:

                    self.hablando = True
                    inicio_detectado = True
                    self.contador_silencio = 0

            else:
                self.contador_voz = 0

        else:

            if rms < self.umbral_fin:

                self.contador_silencio += 1

                if self.contador_silencio >= self.bloques_silencio:
                    self.hablando = False
                    fin_detectado = True
                    self.contador_silencio = 0
                    self.contador_voz = 0

            else:
                self.contador_silencio = 0

        return rms, inicio_detectado, fin_detectado
