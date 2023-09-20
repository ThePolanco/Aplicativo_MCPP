#-Nombrre                -    sigla en rp5      -     sigla en estación udec
#
#-temperatura del aire   -    (T)               -     T
#-Presion atmosferica    -    (Po)              -     preción 
#-Humedad relativa       -    (U)               -     C RH
#-Velocidad de viento    -    (fF)              -     anemometro
#Calidad del aire        -    No hay             -     AirQuality
#-Radiación solar        -    No hay             -     ldr
#-Precipitacion          -    (RRR)             -     No hay -


'''
Valores de rp5 y utilizados para el estudio que ttambien se tienen en la estaación udec
-Presion atmosferica(Po) - preción 
-Humedad relativa(U) - C RH
-Velocidad de viento (fF) - anemometro
-temperatura del aire (T)-T
'''

'''
varriables que se ttendran en cuenta paraa el estudio
-Presion atmosferica(Po) - preción 
-Humedad relativa(U) - C RH
-Velocidad de viento (Ff) - anemometro
-temperatura del aire (T)-T
-Precipitación paara el entrenamiento ya que representa el resultado

'''



class Precipitacion:
    def __init__(self, fecha, hora, Po,T,U,Ff,RRR):
        self.fecha = fecha
        self.hora = hora
        self.Po = Po
        self.T = T
        self.U= U
        self.Ff = Ff
        self.RRR= RRR

    # Metodo para almacenar los documentos
    def formato_doc(self):
        return{
            'fecha':self.fecha,
            'hora':self.hora,
            'Po':self.Po,
            'T':self.T,
            'U':self.U,
            'Ff':self.Ff,
            'RRR':self.RRR
        }