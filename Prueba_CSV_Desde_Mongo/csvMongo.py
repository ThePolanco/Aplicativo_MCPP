'''
Nogales, R. (2022, 30 marzo). Exportando datos a CSV/Excel desde MongoDB con Python. Medium. https://xrnogales.medium.com/exportando-datos-a-csv-excel-desde-mongodb-con-python-f58db58e764f
'''
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, time

#pip install openpyxl


mongo_uri = f'mongodb+srv://Lupo:precipitacionUDEC@cluster0.0s3yt3s.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp' # Your connection uri
client = MongoClient(mongo_uri)
# Se utiliza una base de datos de prueba con solo valores numericos
db = client.get_database('otra')
collection = db.get_collection('a')
# Obtener un df con la colección completa:
df = pd.DataFrame(list(collection.find()))
'''
# Obtener un df con datos filtrados: (precio < 50)
df = pd.DataFrame(list(collection.find({"price": {"$lt": 50}})))
# Obtener un df para los datos de hoy, sin el campo '_id':
today = datetime.today()
midnight = datetime.combine(today, time(0, 0))
df = pd.DataFrame(list(collection.find({"date": { "$gt" : midnight}}, {"_id": 0})))
# "date" field must exists (date type)
'''
## Exportar a Excel o CSV:
df.to_csv('filename.csv', index=False)
df.to_excel('filename.xlsx', index=False)