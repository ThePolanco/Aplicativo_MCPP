from pymongo import MongoClient
import certifi
from pymongo.collection import ReturnDocument
from pymongo import MongoClient

# Conexión con MongoDB
MONGO = 'mongodb+srv://Lupo:precipitacionUDEC@cluster0.0s3yt3s.mongodb.net/?retryWrites=true&w=majority'

# Utilización del certificado
certificado = certifi.where()

# Función para la conexión con la DB
def Conexion():
    try:
        client = MongoClient(MONGO, tlsCAFile = certificado)
        bd = client["bd_Precipitaciones"]
    except ConnectionError:
        print("Error de Conexión")
    return bd

