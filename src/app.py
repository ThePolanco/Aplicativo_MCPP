# Importar Flask y request
from flask import Flask, render_template, request, redirect, url_for

# Creación de la aplicación
app = Flask(__name__)

# Ruta para el index
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la pantalla de probar modelo
@app.route('/prueba')
def prueba():
    return render_template('prueba.html')

#Ruta para la pantalla de datos
@app.route('/datos')
def datos():
    return render_template('datos.html')


# Control del error 404
def error_404(error):
    return render_template('error_404.html'), 404



if __name__ == '__main__':
    app.register_error_handler(404, error_404)
    app.run(debug = True, port = 2023)