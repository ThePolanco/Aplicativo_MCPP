'''
Videos de guia
https://www.youtube.com/watch?v=N1h5b2JGiII
https://www.google.com/search?q=como+exportar+datos+de+mongodb+a+una+archivo+csv&client=opera-gx&hs=GB8&sca_esv=568744667&sxsrf=AM9HkKlxbKNJZqVv6fCUdpOGHDZBQrK-_A%3A1695798065858&ei=MdMTZbyHNICzkvQPmZmNoAQ&ved=0ahUKEwj8482KnMqBAxWAmYQIHZlMA0QQ4dUDCBA&uact=5&oq=como+exportar+datos+de+mongodb+a+una+archivo+csv&gs_lp=Egxnd3Mtd2l6LXNlcnAiMGNvbW8gZXhwb3J0YXIgZGF0b3MgZGUgbW9uZ29kYiBhIHVuYSBhcmNoaXZvIGNzdjIFEAAYogQyBRAAGKIEMgUQABiiBDIFEAAYogRI_ilQ6AVYlh9wAXgBkAEAmAGzAaAB0wqqAQMwLji4AQPIAQD4AQHCAgoQABhHGNYEGLAD4gMEGAAgQYgGAZAGCA&sclient=gws-wiz-serp
https://xrnogales.medium.com/exportando-datos-a-csv-excel-desde-mongodb-con-python-f58db58e764f

'''


import os
# Importar Flask y request
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from config import *
from precipitacion import Precipitacion
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


# Creación de la aplicación
app = Flask(__name__)
# Carpeta de subida del archivo
app.config['UPLOAD_FOLDER'] = './src/Archivos csv'

# Ruta para el index
@app.route('/')
def index():
    # predicciones con el nuevo modelo con recolección de datos desde una DB Remota de mongo

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
    ## Exportar a CSV:
    df.to_csv('./src/ArchivoCSV_Mongo/filename.csv', index=False)

    
    # Se carga el conjunto de datos spam.data 
## sep== separacion por comas
## Header== no tiene
    df=pd.read_csv("./src/ArchivoCSV_Mongo/filename.csv", header=None, skiprows=1)
# separa los datos en "XRF" y en "YRF"
# "XRF" representa todas las columnas menos la ultima la cual es la de respuesta
# "YRF" representa la columna de respuesta
    XRF = df.iloc[:,:-1]
    YRF = df.iloc[:,-1]

# Normalizamos los datos con logaritmos neperianos
    XRF_normalizada=np.log1p(df.iloc[:,:-1])

# configuracion dl modelo con datos normalizados mediante a logaritmos neperianos
    XRF_train, XRF_test, YRF_train, YRF_test = train_test_split(XRF_normalizada,YRF, test_size=0.2, random_state=1)

#Random forest

    rf= RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=1000,min_samples_split=3,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='sqrt',max_leaf_nodes=10,min_impurity_decrease=0.0,bootstrap=True,oob_score=False,n_jobs=-1,random_state=1,verbose=0,warm_start=False,ccp_alpha=0.0,max_samples=10)
    # Ajustar parametros de regresión lineal de datos
    rf.fit(XRF_train,YRF_train)

    YRF_prdss = rf.predict(XRF_test)

#Indice de ocurrencia
    ocurrenciaRF=accuracy_score(YRF_test, YRF_prdss)


# se crea un dataframe para comparar con lso valores reales

    rtaRF = pd.DataFrame({'real': YRF_test,'predicciones': YRF_prdss})


    lista=[YRF_test]

    # *******************************************************************************************************************************************************************
# Support Vector Machines
    datos=pd.read_csv('./src/ArchivoCSV_Mongo/filename.csv', header=None, skiprows=1)

# separa los datos en "XMVS" y en "YMVS"
# "XMVS" representa todas las columnas menos la ultima la cual es la de respuesta
# "YMVS" representa la columna de respuesta
    XMVS = datos.iloc[:,:-1]
    YMVS = datos.iloc[:,-1]

# División de los datos en un 80% para entrenamiento y un 20% para prueba
    XMVS_train,XMVS_test, YMVS_train, YMVS_test = train_test_split(XMVS, YMVS, test_size=0.2, random_state=42)

    clf = SVC(kernel = 'linear').fit(XMVS_train, YMVS_train)
    ocurrenciaMVS=clf.score(XMVS_test, YMVS_test)


    YMVS_pred = clf.predict(XMVS_test)

# Calculo de la matriz de correlación
    correlation_matrix = datos.corr()

    RTAMVS = pd.DataFrame({'Real': YMVS_test,'Predicho': YMVS_pred})


# ***************************************************************************************************************************************



# Modelo Naive Bayes


# separa los datos en "XNV" y en "Y"
# "XNV" representa todas las columnas menos la ultima la cual es la de respuesta
# "Y" representa la columna de respuesta


    df_data = pd.read_csv('./src/ArchivoCSV_Mongo/filename.csv', header=None, skiprows=1)

# Haremos Feature Selection para mejorar los resultados del algoritmo. Utilizando la clase SelectKBest para seleccionar las 5 mejores caracteristicas. Son las variables que mas aportan al momento de realizar la clasificación.

    XNV = df_data.iloc[:,:-1]
    y = df_data.iloc[:,-1]

    best = SelectKBest(k = 4)
    XNV_new = best.fit_transform(XNV,y)
    XNV_new.shape
    selected = best.get_support(indices = True)

    used_features = XNV.columns[selected]

# División de los datos en conjuntos de prueba y entrenamiento, 80% para entrenamiento y 20% para prueba

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 6)
    XNV_train, XNV_test, y_train, y_test = train_test_split(XNV,y, test_size = 0.2, random_state = 6)

# Normalizacion de la data

    bayes_naive = GaussianNB()
    bayes_clasif = bayes_naive.fit(XNV_train[used_features].values, y_train)
    y_pred = bayes_naive.predict(XNV_test[used_features])
#y_pred = bayes_clasif.predict(X_test)

#Indice de ocurrencia
    ocurrenciaNV=accuracy_score(y_test, y_pred)

    RTANV = pd.DataFrame({'Real': y_test,'Predicho': y_pred})

    resultadosDb = {
        'realRF':YRF_test,
        'prediccionesRF':YRF_prdss,
        'PorcentajeRF':ocurrenciaRF,
        'realMVS':YMVS_test,
        'prediccionesMVS':YMVS_pred,
        'PorcentajeMVS':ocurrenciaMVS,
        'realNV':y_test,
        'prediccionesNV':y_pred,
        'PorcentajeNV':ocurrenciaNV
        }
    #Comparacion de los modelos con el modelo Lupo
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Guardar la cantidad de resultados generados por los algoritmos
    TamaA=len(YRF_prdss)
    #Crear un bucle Que permita leer cada dato
    lupo = []
    ni=0
    #Configuración del grado de ocurrencia tolerado para la comparacion de lupo
    ocurrenciaAdimitica=0.65
    while ni<TamaA:
        #Comparar ocurrencia para generar nuevos resultados
        # RF=1 MVS=1 NV=1
        if(ocurrenciaRF>ocurrenciaAdimitica and ocurrenciaMVS>ocurrenciaAdimitica and ocurrenciaNV>ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(0)
            else:
                print("no concuerda")
        # RF=0 MVS=1 NV=1
        elif(ocurrenciaRF<ocurrenciaAdimitica and ocurrenciaMVS>ocurrenciaAdimitica and ocurrenciaNV>ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                if(ocurrenciaMVS>ocurrenciaNV):
                    lupo.append(0)
                else:
                    lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                if(ocurrenciaMVS>ocurrenciaNV):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                if(ocurrenciaMVS>ocurrenciaNV):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                if(ocurrenciaMVS>ocurrenciaNV):
                    lupo.append(0)
                else:
                    lupo.append(1)
            else:
                print("no concuerda")
        # RF=1 MVS=0 NV=1
        elif(ocurrenciaRF>ocurrenciaAdimitica and ocurrenciaMVS<ocurrenciaAdimitica and ocurrenciaNV>ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                if(ocurrenciaRF>ocurrenciaNV):
                    lupo.append(0)
                else:
                    lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                if(ocurrenciaRF>ocurrenciaNV):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                if(ocurrenciaRF>ocurrenciaNV):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                if(ocurrenciaRF>ocurrenciaNV):
                    lupo.append(0)
                else:
                    lupo.append(1)
        # RF=1 MVS=1 NV=0
        elif(ocurrenciaRF>ocurrenciaAdimitica and ocurrenciaMVS>ocurrenciaAdimitica and ocurrenciaNV<ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                if(ocurrenciaRF>ocurrenciaMVS):
                    lupo.append(0)
                else:
                    lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                if(ocurrenciaRF>ocurrenciaMVS):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                if(ocurrenciaRF>ocurrenciaMVS):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                if(ocurrenciaRF>ocurrenciaMVS):
                    lupo.append(0)
                else:
                    lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(0)
            else:
                print("no concuerda")
        # RF> MVS and NV
        elif(ocurrenciaRF>ocurrenciaAdimitica and ocurrenciaMVS<ocurrenciaAdimitica and ocurrenciaNV<ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(0)
            else:
                print("no concuerda")
        # MVS> RF and NV
        elif(ocurrenciaRF<ocurrenciaAdimitica and ocurrenciaMVS>ocurrenciaAdimitica and ocurrenciaNV<ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(0)
            else:
                print("no concuerda")
        # NV> RF and MVS
        elif(ocurrenciaRF<ocurrenciaAdimitica and ocurrenciaMVS<ocurrenciaAdimitica and ocurrenciaNV>ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(1)
            else:
                print("no concuerda")
        #else final
        else:
            print("No concuerda el valor")
        ni=ni+1
    #Utilizacion del documento para extraer datos de fecha y hora
    # predicciones con el nuevo modelo con recolección de datos desde una DB Remota de mongo
    BusquedaMongo = f'mongodb+srv://Lupo:precipitacionUDEC@cluster0.0s3yt3s.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp' # Your connection uri
    cliente = MongoClient(BusquedaMongo)
    # Se utiliza una base de datos de prueba con solo valores numericos
    bd = cliente.get_database('bd_Precipitaciones')
    coleccion = bd.get_collection('Datos')
    # Obtener un DataF con la colección completa:
    DataF = pd.DataFrame(list(coleccion.find()))
    DataF.drop('T', inplace=True, axis=1)
    DataF.drop('Po', inplace=True, axis=1)
    DataF.drop('U', inplace=True, axis=1)
    DataF.drop('Ff', inplace=True, axis=1)
    DataF.drop('RRR', inplace=True, axis=1)
    DataF.drop('_id', inplace=True, axis=1)
    DataF.to_csv('./src/fechayhora/FechayHora.csv', index=False)
    #Ocurrencia de Lupo
    ocurrenciaLupo=accuracy_score(YMVS_test, lupo)
    #muestreo de datos del ultimo dato
    FeyHo = pd.read_csv('./src/fechayhora/FechayHora.csv', header=None, skiprows=1)
    Ffecha = FeyHo.iloc[0,0]
    Hhora = FeyHo.iloc[0,1]
    # Eleccion del algorimo que va a dar el resultado de la predicción


    lluvia=""
    # Lupo>RF and MVS and NV
    if(ocurrenciaLupo>ocurrenciaRF and ocurrenciaLupo>ocurrenciaMVS and ocurrencialupo>ocurrenciaNV):
        if(lupo[TamaA-1]==1):
            lluvia="llovera"
        else:
            lluvia="No llovera"
    # RF>Lupo and MVS and NV
    elif(ocurrenciaRF>ocurrenciaLupo and ocurrenciaRF>ocurrenciaMVS and ocurrenciaRF>ocurrenciaNV):
        if(YRF_prdss[TamaA-1]==1):
            lluvia="llovera"
        else:
            lluvia="No llovera"
    # MVS>Lupo and RF and NV
    elif(ocurrenciaMVS>ocurrenciaLupo and ocurrenciaMVS>ocurrenciaRF and ocurrenciaMVS>ocurrenciaNV):
        if(YMVS_pred[TamaA-1]==1):
            lluvia="llovera"
        else:
            lluvia="No llovera"
    # NV>Lupo and RF and MVS
    elif(ocurrenciaNV>ocurrenciaLupo and ocurrenciaNV>ocurrenciaRF and ocurrenciaNV>ocurrenciamVS):
        if(y_pred[TamaA-1]==1):
            lluvia="llovera"
        else:
            lluvia="No llovera"
    # NV==Lupo == RF == MVS
    else:
        if(YMVS_pred[TamaA-1]==1):
            lluvia="llovera"
        else:
            lluvia="No llovera"

    resultadosDb = {
        'realRF':YRF_test,
        'prediccionesRF':YRF_prdss,
        'PorcentajeRF':ocurrenciaRF,
        'realMVS':YMVS_test,
        'prediccionesMVS':YMVS_pred,
        'PorcentajeMVS':ocurrenciaMVS,
        'realNV':y_test,
        'prediccionesNV':y_pred,
        'PorcentajeNV':ocurrenciaNV,
        'LupoRTA':lupo,
        'LupoOcurrencia':ocurrenciaLupo,
        'lluvia':lluvia,
        'FFecha': Ffecha,
        'HHora':Hhora
        }

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #return "<h2> {%lista%} </h2>"
    return render_template('index.html',Predicciones=resultadosDb)


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#*****************************************************************************************************************************************************************************************************

# Ruta para la pantalla de probar modelo
@app.route('/prueba')
def prueba():    
    return render_template('prueba.html')
# Funcion para recibir el archivo y enviarlo a una carpeta llamada Archivos csv creada en la carpeta src
#Aqui se hara todo el proceso para leer el archivo seleccionado y aplicaar los modelos
@app.route('/RTA', methods=['POST'])
def procesoAlgoritmo():
   if request.method == 'POST':
  # obtenemos el archivo del input "archivo"
    f = request.files['archivo']
    filename = secure_filename(f.filename)
  # Guardamos el archivo en el directorio "Archivos PDF"
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))



#test_size- valor de entrenamiento

    traine=int(request.form['test_size'])
    trainee=traine*0.01
# Random forest


# Se carga el conjunto de datos spam.data 
## sep== separacion por comas
## Header== no tiene
    df=pd.read_csv("./src/Archivos csv/archivo.csv", sep=',', header=None)
# separa los datos en "XRF" y en "YRF"
# "XRF" representa todas las columnas menos la ultima la cual es la de respuesta
# "YRF" representa la columna de respuesta
    XRF = df.iloc[:,:-1]
    YRF = df.iloc[:,-1]

# Normalizamos los datos con logaritmos neperianos
    XRF_normalizada=np.log1p(df.iloc[:,:-1])

# configuracion dl modelo con datos normalizados mediante a logaritmos neperianos
    XRF_train, XRF_test, YRF_train, YRF_test = train_test_split(XRF_normalizada,YRF, test_size=trainee, random_state=1)

#Random forest

    rf= RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=1000,min_samples_split=3,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='sqrt',max_leaf_nodes=10,min_impurity_decrease=0.0,bootstrap=True,oob_score=False,n_jobs=-1,random_state=1,verbose=0,warm_start=False,ccp_alpha=0.0,max_samples=10)

    rf.fit(XRF_train,YRF_train)

    YRF_prdss = rf.predict(XRF_test)

#Indice de ocurrencia
    ocurrenciaRF=accuracy_score(YRF_test, YRF_prdss)
    print("El porcentaje de efectividad de random forest es: ", ocurrenciaRF)


# se crea un dataframe para comparar con lso valores reales
    print("* Datos de Randon Forest")
    rtaRF = pd.DataFrame({'real': YRF_test,'predicciones': YRF_prdss})


    lista=[YRF_test]

    print(rtaRF)

    # *******************************************************************************************************************************************************************
# Support Vector Machines
    datos=pd.read_csv('./src/Archivos csv/archivo.csv', sep=',', header=None)

# separa los datos en "XMVS" y en "YMVS"
# "XMVS" representa todas las columnas menos la ultima la cual es la de respuesta
# "YMVS" representa la columna de respuesta

    YMVS = datos.iloc[:,-1]
    XMVS = datos.iloc[:,:-1]



# División de los datos en un 80% para entrenamiento y un 20% para prueba
    XMVS_train,XMVS_test, YMVS_train, YMVS_test = train_test_split(XMVS, YMVS, test_size=trainee, random_state=42)

    clf = SVC(kernel = 'linear').fit(XMVS_train, YMVS_train)
    ocurrenciaMVS=clf.score(XMVS_test, YMVS_test)
    print("El porcentaje de efectividad del modelo Support Vector Machines es: ", ocurrenciaMVS)

    YMVS_pred = clf.predict(XMVS_test)

# Calculo de la matriz de correlación
    correlation_matrix = datos.corr()

    print("* Datos de Support Vector Machines")
    RTAMVS = pd.DataFrame({'Real': YMVS_test,'Predicho': YMVS_pred})

    print(RTAMVS)


# ***************************************************************************************************************************************



# Modelo Naive Bayes


# separa los datos en "XNV" y en "Y"
# "XNV" representa todas las columnas menos la ultima la cual es la de respuesta
# "Y" representa la columna de respuesta


    df_data = pd.read_csv('./src/Archivos csv/archivo.csv', sep=',', header=None)

# Haremos Feature Selection para mejorar los resultados del algoritmo. Utilizando la clase SelectKBest para seleccionar las 5 mejores caracteristicas. Son las variables que mas aportan al momento de realizar la clasificación.

    XNV = df_data.iloc[:,:-1]
    y = df_data.iloc[:,-1]

    best = SelectKBest(k = 5)
    XNV_new = best.fit_transform(XNV,y)
    XNV_new.shape
    selected = best.get_support(indices = True)

    used_features = XNV.columns[selected]

# División de los datos en conjuntos de prueba y entrenamiento, 80% para entrenamiento y 20% para prueba

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 6)
    XNV_train, XNV_test, y_train, y_test = train_test_split(XNV,y, test_size = trainee, random_state = 6)


# Normalizacion de la data

    bayes_naive = GaussianNB()
    bayes_clasif = bayes_naive.fit(XNV_train[used_features].values, y_train)
    y_pred = bayes_naive.predict(XNV_test[used_features])
#y_pred = bayes_clasif.predict(X_test)

#Indice de ocurrencia
    ocurrenciaNV=accuracy_score(y_test, y_pred)

    print("El porcentaje de efectividad de Naive Bayes es: ", ocurrenciaNV)

    RTANV = pd.DataFrame({'Real': y_test,'Predicho': y_pred})

    print(RTANV)




    #Comparacion de los modelos con el modelo Lupo
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Guardar la cantidad de resultados generados por los algoritmos
    TamaA=len(YRF_prdss)
    #Crear un bucle Que permita leer cada dato
    lupo = []
    ni=0
    #Configuración del grado de ocurrencia tolerado para la comparacion de lupo
    ocurrenciaAdimitica=0.60
    while ni<TamaA:
        #Comparar ocurrencia para generar nuevos resultados
        # RF=1 MVS=1 NV=1
        if(ocurrenciaRF>ocurrenciaAdimitica and ocurrenciaMVS>ocurrenciaAdimitica and ocurrenciaNV>ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(0)
            else:
                print("no concuerda")
        # RF=0 MVS=1 NV=1
        elif(ocurrenciaRF<ocurrenciaAdimitica and ocurrenciaMVS>ocurrenciaAdimitica and ocurrenciaNV>ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                if(ocurrenciaMVS>ocurrenciaNV):
                    lupo.append(0)
                else:
                    lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                if(ocurrenciaMVS>ocurrenciaNV):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                if(ocurrenciaMVS>ocurrenciaNV):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                if(ocurrenciaMVS>ocurrenciaNV):
                    lupo.append(0)
                else:
                    lupo.append(1)
            else:
                print("no concuerda")
        # RF=1 MVS=0 NV=1
        elif(ocurrenciaRF>ocurrenciaAdimitica and ocurrenciaMVS<ocurrenciaAdimitica and ocurrenciaNV>ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                if(ocurrenciaRF>ocurrenciaNV):
                    lupo.append(0)
                else:
                    lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                if(ocurrenciaRF>ocurrenciaNV):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                if(ocurrenciaRF>ocurrenciaNV):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                if(ocurrenciaRF>ocurrenciaNV):
                    lupo.append(0)
                else:
                    lupo.append(1)
        # RF=1 MVS=1 NV=0
        elif(ocurrenciaRF>ocurrenciaAdimitica and ocurrenciaMVS>ocurrenciaAdimitica and ocurrenciaNV<ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                if(ocurrenciaRF>ocurrenciaMVS):
                    lupo.append(0)
                else:
                    lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                if(ocurrenciaRF>ocurrenciaMVS):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                if(ocurrenciaRF>ocurrenciaMVS):
                    lupo.append(1)
                else:
                    lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                if(ocurrenciaRF>ocurrenciaMVS):
                    lupo.append(0)
                else:
                    lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(0)
            else:
                print("no concuerda")
        # RF> MVS and NV
        elif(ocurrenciaRF>ocurrenciaAdimitica and ocurrenciaMVS<ocurrenciaAdimitica and ocurrenciaNV<ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(0)
            else:
                print("no concuerda")
        # MVS> RF and NV
        elif(ocurrenciaRF<ocurrenciaAdimitica and ocurrenciaMVS>ocurrenciaAdimitica and ocurrenciaNV<ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(0)
            else:
                print("no concuerda")
        # NV> RF and MVS
        elif(ocurrenciaRF<ocurrenciaAdimitica and ocurrenciaMVS<ocurrenciaAdimitica and ocurrenciaNV>ocurrenciaAdimitica):
            if(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(1)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==1 and YMVS_pred[ni]==0 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==1 and y_pred[ni]==0):
                lupo.append(0)
            elif(YRF_prdss[ni]==0 and YMVS_pred[ni]==0 and y_pred[ni]==1):
                lupo.append(1)
            else:
                print("no concuerda")
        #else final
        else:
            print("No concuerda el valor")
        ni=ni+1
    
    #Ocurrencia de Lupo
    ocurrenciaLupo=accuracy_score(YMVS_test, lupo)


    resultados = {
        'realRF':YRF_test,
        'prediccionesRF':YRF_prdss,
        'PorcentajeRF':ocurrenciaRF,
        'realMVS':YMVS_test,
        'prediccionesMVS':YMVS_pred,
        'PorcentajeMVS':ocurrenciaMVS,
        'realNV':y_test,
        'prediccionesNV':y_pred,
        'PorcentajeNV':ocurrenciaNV,
        'Modelolupo':lupo,
        'OcurrenciaLupo':ocurrenciaLupo
        }
    #return "<h2> {%lista%} </h2>"
    return render_template('ResultadoModelo.html', datos=resultados)


#*****************************************************************************************************************************************************************************************************
con_bd = Conexion()
#Ruta para la pantalla de datos donde se muestra toda la Data
@app.route('/datos')
def datos():
    precipitaciones = con_bd['Datos']
    PrecipitacionesRegistradas=precipitaciones.find()
    return render_template('datos.html', precipitaciones = PrecipitacionesRegistradas)

#Ruta para la pantalla de datos donde se muestra la data consultada
@app.route('/fechaBuscada',methods = ['POST'])
def Read():
    precipitaciones = con_bd['Datos']
    fechabuscada = request.form['fecha']
    query={"fecha":fechabuscada}
    PrecipitacionesRegistradas=precipitaciones.find(query)
    return render_template('datos.html', precipitaciones = PrecipitacionesRegistradas)


# inicializacion de una variable publica que captura el archivo inserttado por el usuario
ArchivoG=pd.read_csv('./src/Archivos csv/archivo.csv', sep=',', header=None)


# Creación de un Grafico de Matriz de correlacion con el archivo subido por el usuario
@app.route('/Mcorrelacion')
def Mcorrelacion():
    correlation_matrix = ArchivoG.corr()
# Creación de un mapa de calor de la matriz de correlación
    plt.figure(figsize=(10, 8))  # Tamaño de la figura
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Matriz de Correlación")
    plt.savefig("./src/Archivos img/GraficoMC.jpg")
    plt.show()
    return redirect(url_for('index'))


# Creación de un Grafico de todas las variables con el archivo subido por el usuario
@app.route('/GraficoDT')
def GraficoDT():
    ArchivoG.hist()
    plt.savefig("./src/Archivos img/GraficoDT.jpg")
    plt.show()
    return redirect(url_for('index'))

# Control del error 404
def error_404(error):
    return render_template('error_404.html'), 404



if __name__ == '__main__':
    app.register_error_handler(404, error_404)
    app.run(debug = True, port = 2023)