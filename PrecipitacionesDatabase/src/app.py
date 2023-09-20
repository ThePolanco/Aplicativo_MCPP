from flask import Flask, redirect, render_template, request, url_for
from config import *
from precipitacion import Precipitacion


# Instancias para realizar operaciones con la DB
con_bd = Conexion()

app = Flask(__name__)

@app.route('/')
def index():
    # Se modifica la vista index para poder hacer el muestreo de los datos
    precipitaciones = con_bd['Datos']
    PrecipitacionesRegistradas=precipitaciones.find()
    return render_template('index.html', precipitaciones = PrecipitacionesRegistradas)

# Ruta para guardar los datos de la DB
@app.route('/guardar_datos', methods = ['POST'])
def agregarDatos():
    precipitaciones= con_bd['Datos']
    fecha = request.form['fecha']
    hora = request.form['hora']
    Po = request.form['Po']
    T = request.form['T']
    U= request.form['U']
    Ff = request.form['Ff']
    RRR= request.form['RRR']

    

    if fecha and hora and Po and T and U and Ff and RRR:
        precipitacion = Precipitacion(fecha, hora, Po,T,U,Ff,RRR)
        #insert_one para crear un documento en Mongo
        precipitaciones.insert_one(precipitacion.formato_doc())
        return redirect(url_for('index'))
    else:
        return "Error"
    

# En este caso se eliminara atravez de la URL
# Ruta para eliminar datos en la DB donde la ruta se llama eliminar_persona y recibe un parametro llamado nombre_persona
@app.route('/eliminar_fecha/<string:fecha_Precipitacion>')
def eliminar(fecha_Precipitacion):
    precipitaciones = con_bd['Datos']
    # Se hace uso de delete_one para borrar los datos de la DB datos donde el dato que se elimina es el que se para como argumento para nombre
    precipitaciones.delete_one({ 'fecha': fecha_Precipitacion})
    # Creamos un redireccionamiento que redirija a la vista index
    return redirect(url_for('index'))

#Editar o actualizar el contenido 
@app.route('/editar_dato/<string:fecha_Precipitacion>', methods = ['POST'])
def editar(fecha_Precipitacion):
    precipitaciones = con_bd['Datos']
    # Se realiza el mismo proceso de inserción y extracción para poder actualizar los datos
    fecha = request.form['fecha']
    hora = request.form['hora']
    Po = request.form['Po']
    T = request.form['T']
    U= request.form['U']
    Ff = request.form['Ff']
    RRR= request.form['RRR']
    # Utilizaremos la función update_one()
    if fecha and hora and Po and T and U and Ff and RRR:
        precipitaciones.update_one({'fecha': fecha_Precipitacion}, 
                            {'$set': {'fecha' : fecha , 'hora': hora, 'Po': Po,'T': T ,'U': U ,'Ff': Ff ,'RRR': RRR}}) # update_one() necesita de al menos dos parametros para funcionar
        return redirect(url_for('index'))
    else:
        return "Error de actualización"


if __name__ == '__main__':
    app.run(debug = True, port = 2001)