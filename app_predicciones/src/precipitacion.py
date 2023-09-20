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