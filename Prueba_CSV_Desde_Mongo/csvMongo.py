'''
Nogales, R. (2022, 30 marzo). Exportando datos a CSV/Excel desde MongoDB con Python. Medium. https://xrnogales.medium.com/exportando-datos-a-csv-excel-desde-mongodb-con-python-f58db58e764f
'''
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, time



# Importar Flask y request
from flask import Flask, render_template, request, redirect, url_for
#


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# Separar conjunto de entrenamiento y de validacion
from sklearn.model_selection import train_test_split
# se empiezan a construir los clasificadores con arboles de decisión 
from sklearn.tree import DecisionTreeClassifier
# Utilizaremos random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from matplotlib import colors
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest





#pip install openpyxl


mongo_uri = f'mongodb+srv://Lupo:precipitacionUDEC@cluster0.0s3yt3s.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp' # Your connection uri
client = MongoClient(mongo_uri)
# Se utiliza una base de datos de prueba con solo valores numericos
db = client.get_database('bd_Precipitaciones')
collection = db.get_collection('Datos')
# Obtener un df con la colección completa:
df = pd.DataFrame(list(collection.find()))
#Eliminar parametros innecesarios
df.drop('fecha', inplace=True, axis=1)
df.drop('hora', inplace=True, axis=1)
df.drop('_id', inplace=True, axis=1)


'''
# Obtener un df con datos filtrados: (precio < 50)
df = pd.DataFrame(list(collection.find({"price": {"$lt": 50}})))
# Obtener un df para los datos de hoy, sin el campo '_id':
today = datetime.today()
midnight = datetime.combine(today, time(0, 0))
df = pd.DataFrame(list(collection.find({"date": { "$gt" : midnight}}, {"_id": 0})))
# "date" field must exists (date type)
'''
## Exportar a CSV:
df.to_csv('filename.csv', index=False)

#**************************************************************************************************************************************************************
#leer archivo quittando lasa primeras filas
ddaa=pd.read_csv("C:/Users/Usuario/Desktop/Tesis/Proyecto_de_Grado/filename.csv", sep=',', header=None)


X = ddaa.iloc[1:,:-1]

print(X)
