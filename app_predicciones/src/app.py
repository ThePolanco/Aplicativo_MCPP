import os
# Importar Flask y request
from flask import Flask, render_template, request, redirect, url_for,session,flash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from config import *
from precipitacion import Precipitacion
from datetime import datetime
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
# Indice de f1_score
from sklearn.metrics import f1_score

# Creación de la aplicación
app = Flask(__name__)
# Carpeta de subida del archivo
app.config['UPLOAD_FOLDER'] = './src/Archivos csv'

# Ruta para el index
@app.route('/')
def index():
    # Predicciones con el nuevo modelo con recolección de datos desde una DB Remota de mongo
    mongo_uri = f'mongodb+srv://Lupo:precipitacionUDEC@cluster0.0s3yt3s.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp' # Your connection uri
    client = MongoClient(mongo_uri)
    # Se utiliza una base de datos de prueba con solo valores numericos
    db = client.get_database('bd_Precipitaciones')
    collection = db.get_collection('Datos')
    # Obtener un df con la colección completa:
    df = pd.DataFrame(list(collection.find()))
    # Eliminar parametros innecesarios
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
# Indice de f1_score
    fl_RF=f1_score(YRF_test, YRF_prdss)
    print("???????????????????????????????????????????????????????????????????????????????????????????????")
    print(fl_RF)
    print("???????????????????????????????????????????????????????????????????????????????????????????????")    


# Se crea un dataframe para comparar con los valores reales

    rtaRF = pd.DataFrame({'real': YRF_test,'predicciones': YRF_prdss})

    lista=[YRF_test]

# *******************************************************
# Support Vector Machines
    datos=pd.read_csv('./src/ArchivoCSV_Mongo/filename.csv', header=None, skiprows=1)

# Separa los datos en "XMVS" y en "YMVS"
# "XMVS" representa todas las columnas menos la ultima la cual es la de respuesta
# "YMVS" representa la columna de respuesta
    XMVS = datos.iloc[:,:-1]
    YMVS = datos.iloc[:,-1]

# División de los datos en un 80% para entrenamiento y un 20% para prueba
    XMVS_train,XMVS_test, YMVS_train, YMVS_test = train_test_split(XMVS, YMVS, test_size=0.2, random_state=42)

    clf = SVC(kernel = 'linear').fit(XMVS_train, YMVS_train)
    YMVS_pred = clf.predict(XMVS_test)
    ocurrenciaMVS=clf.score(XMVS_test, YMVS_test)
    # Indice de f1_score
    fl_MVS = f1_score(YMVS_test, YMVS_pred)
    print("???????????????????????????????????????????????????????????????????????????????????????????????")
    print(fl_MVS)
    print("???????????????????????????????????????????????????????????????????????????????????????????????") 

    YMVS_pred = clf.predict(XMVS_test)

# Calculo de la matriz de correlación
    correlation_matrix = datos.corr()

    RTAMVS = pd.DataFrame({'Real': YMVS_test,'Predicho': YMVS_pred})

# *********************************************

# Modelo Naive Bayes

# Separa los datos en "XNV" y en "Y"
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

    fl_NB=f1_score(y_test, y_pred)
    print("???????????????????????????????????????????????????????????????????????????????????????????????")
    print(fl_NB)
    print("???????????????????????????????????????????????????????????????????????????????????????????????") 


    RTANV = pd.DataFrame({'Real': y_test,'Predicho': y_pred})

# Comparacion de los modelos con el modelo Lupo
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
    # Utilizacion del documento para extraer datos de fecha y hora
    # Predicciones con el nuevo modelo con recolección de datos desde una DB Remota de mongo
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
    # Ocurrencia de Lupo
    ocurrenciaLupo=accuracy_score(YMVS_test, lupo)
    # Indice de f1_score
    fl_LUPO=f1_score(YMVS_test, lupo)
    print("???????????????????????????????????????????????????????????????????????????????????????????????")
    print(fl_LUPO)
    print("???????????????????????????????????????????????????????????????????????????????????????????????") 

    # Muestreo de datos del ultimo dato
    FeyHo = pd.read_csv('./src/fechayhora/FechayHora.csv', header=None, skiprows=1)
    Ffecha = FeyHo.iloc[-1,0]
    Hhora = FeyHo.iloc[-1,1]
    Ifecha = FeyHo.iloc[0,0]
    # Eleccion del algorimo que va a dar el resultado de la predicción


    lluvia=""
    # Lupo>RF and MVS and NV
    if(ocurrenciaLupo>ocurrenciaRF and ocurrenciaLupo>ocurrenciaMVS and ocurrenciaLupo>ocurrenciaNV):
        if(lupo[TamaA-1]==1):
            lluvia="Llovera"
        else:
            lluvia="No llovera"
    # RF>Lupo and MVS and NV
    elif(ocurrenciaRF>ocurrenciaLupo and ocurrenciaRF>ocurrenciaMVS and ocurrenciaRF>ocurrenciaNV):
        if(YRF_prdss[TamaA-1]==1):
            lluvia="Llovera"
        else:
            lluvia="No llovera"
    # MVS>Lupo and RF and NV
    elif(ocurrenciaMVS>ocurrenciaLupo and ocurrenciaMVS>ocurrenciaRF and ocurrenciaMVS>ocurrenciaNV):
        if(YMVS_pred[TamaA-1]==1):
            lluvia="Llovera"
        else:
            lluvia="No llovera"
    # NV>Lupo and RF and MVS
    elif(ocurrenciaNV>ocurrenciaLupo and ocurrenciaNV>ocurrenciaRF and ocurrenciaNV>ocurrenciaMVS):
        if(y_pred[TamaA-1]==1):
            lluvia="Llovera"
        else:
            lluvia="No llovera"
    # NV==Lupo == RF == MVS
    else:
        if(YMVS_pred[TamaA-1]==1):
            lluvia="Llovera"
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
        'IFecha': Ifecha,
        'HHora':Hhora,
        'Fl_RF':fl_RF,
        'Fl_MVS':fl_MVS,
        'Fl_NB':fl_NB,
        'Fl_LUPO':fl_LUPO
        }
        

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Return "<h2> {%lista%} </h2>"
    # Visualización de los ultimos diez datos utilizados en laas predicciones
    precipitaciones = con_bd['Datos']
    # Limita la consulta a los últimos 10 registros
    PrecipitacionesRegistradas = precipitaciones.find()
    

    return render_template('index.html',Predicciones=resultadosDb,precipitaciones=PrecipitacionesRegistradas)



#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

# Ruta para la pantalla de probar modelo
@app.route('/prueba')
def prueba():    
    return render_template('prueba.html')
# Funcion para recibir el archivo y enviarlo a una carpeta llamada 
# Archivos csv creada en la carpeta src
# Aqui se hara todo el proceso para leer el archivo seleccionado y aplicar los modelos
@app.route('/RTA', methods=['POST'])
def procesoAlgoritmo():
   if request.method == 'POST':
  # Obtenemos el archivo del input "archivo"
    f = request.files['archivo']
    filename = secure_filename(f.filename)
  # Guardamos el archivo en el directorio "Archivos PDF"
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
# test_size- valor de entrenamiento
    traine=int(request.form['test_size'])
    trainee=traine*0.01
    
    # Random forest

# Se carga el conjunto de datos spam.data 
# sep== separacion por comas
# Header== no tiene
    df=pd.read_csv("./src/Archivos csv/archivo.csv", sep=',', header=None)
# Separa los datos en "XRF" y en "YRF"
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

    print("El porcentaje de efectividad de Random Forest es: ", ocurrenciaRF)


# se crea un dataframe para comparar con lso valores reales
    print("* Datos de Randon Forest")
    rtaRF = pd.DataFrame({'real': YRF_test,'predicciones': YRF_prdss})


    lista=[YRF_test]

    print(rtaRF)

    # *******************************************************
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
    print("El porcentaje de efectividad del modelo Support Vector Machine es: ", ocurrenciaMVS)

    YMVS_pred = clf.predict(XMVS_test)

# Calculo de la matriz de correlación
    correlation_matrix = datos.corr()

    print("* Datos de Support Vector Machines")
    RTAMVS = pd.DataFrame({'Real': YMVS_test,'Predicho': YMVS_pred})

    print(RTAMVS)


# *********************************************

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


    # Comparacion de los modelos con el modelo Lupo
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
                print("No concuerda")
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
                print("No concuerda")
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
                print("No concuerda")
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
                print("No concuerda")
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
                print("No concuerda")
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
                print("No concuerda")
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


#*******************************************************************
con_bd = Conexion()

#Ruta para la pantalla de CRUD de Mongo donde se muestra la data consultada
@app.route('/fechaDBBuscada',methods = ['POST'])
def fechaDBBuscada():
    precipitaciones = con_bd['Datos']
    fechabuscada = request.form['fecha']
    query={"fecha":fechabuscada}
    PrecipitacionesRegistradas=precipitaciones.find(query)
    return render_template('datos.html', precipitaciones = PrecipitacionesRegistradas)
    
# Inicializacion de una variable publica que captura el archivo inserttado por el usuario
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

# Creación de un Grafico de Matriz de correlacion con el archivo de la DB MONGO
@app.route('/McorrelacionMongo')
def McorrelacionMongo():
    DFMONGO=pd.read_csv("./src/ArchivoCSV_Mongo/filename.csv", header=None, skiprows=1)
    correlation_matrix = DFMONGO.corr()
# Creación de un mapa de calor de la matriz de correlación
    plt.figure(figsize=(10, 8))  # Tamaño de la figura
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Matriz de Correlación")
    plt.savefig("./src/IMG MONGO/GraficoMC.jpg")
    plt.show()
    return redirect(url_for('index'))

# Creación de un grafico con el balance de los resultados de la data en la DB
@app.route('/GbalanceMongo')
def GbalancenMongo():
    DFMONGO = pd.read_csv("./src/ArchivoCSV_Mongo/filename.csv", header=None, skiprows=1)

    XRF = DFMONGO.iloc[:, :-1]
    YRF = DFMONGO.iloc[:, -1]

    # Contar la cantidad de muestras en cada clase
    class_counts = YRF.value_counts()
    
    # Crear un gráfico de barras para mostrar el balance de clases
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar')
    plt.title('Balance de Clases')
    plt.xlabel('Clase')
    plt.ylabel('Cantidad de Muestras')

    plt.savefig("./src/IMG MONGO/balance_chart.png")
    plt.show()

    return redirect(url_for('index'))


app.secret_key = 'modeloMCPPUDEC'

# Ruta para Login 
@app.route('/login')
def usuario():
    return render_template('login.html')

#Validación de usuario
@app.route('/validar', methods=['POST'])
def validar():
    usuario = request.form['usuario']
    password = request.form['password']

    usuarios = con_bd['Usuarios']
    user_data = usuarios.find_one({"usuario": usuario, "password": password})

    if user_data:
        # Autenticación exitosa, almacenar el usuario en la sesión
        session['usuario'] = usuario
        return redirect(url_for('inicioDB'))
    else:
        flash('Error de autenticación, Usuario o Contraseña incorrecta', 'error')
        return redirect(url_for('usuario'))


@app.route('/inicioDB', methods=['GET', 'POST'])
def inicioDB():
    if 'usuario' in session:
        precipitaciones = con_bd['Datos']

        if request.method == 'POST':
            # Si se envió el formulario, obtener la cantidad de datos deseada
            cantidad_datos = int(request.form.get('cantidadDatos', 30))
            # Ajustar la consulta a la base de datos para obtener la cantidad de datos especificada
            PrecipitacionesRegistradas = precipitaciones.find().sort('fecha', -1).limit(cantidad_datos)
        else:
            # Si no se envió el formulario, mostrar los últimos 50 registros por defecto
            PrecipitacionesRegistradas = precipitaciones.find().sort('fecha', -1).limit(50)

        return render_template('inicioDB.html', precipitaciones=PrecipitacionesRegistradas)
    else:
        return redirect(url_for('usuario'))
    

@app.route('/buscar_por_fecha', methods=['POST'])
def buscar_por_fecha():
    if 'usuario' in session:
        precipitaciones = con_bd['Datos']

        fecha_busqueda = request.form.get('fechaBusqueda')

        try:
            # Ajustar la consulta para buscar registros con la fecha proporcionada
            # en el formato que estás utilizando
            PrecipitacionesRegistradas = precipitaciones.find({'fecha': fecha_busqueda}).sort('fecha', -1)
        except ValueError:
            flash('Formato de fecha incorrecto. Use el formato AAAA-MM-DD.', 'error')
            return redirect(url_for('inicioDB'))

        return render_template('inicioDB.html', precipitaciones=PrecipitacionesRegistradas)
    else:
        return redirect(url_for('usuario'))


@app.route('/logout')
def logout():
    # Eliminar el usuario de la sesión y redirigir al login
    session.pop('usuario', None)
    return redirect(url_for('usuario'))

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

    Po = float(Po)
    T = float(T)
    U= float(U)
    Ff = float(Ff)

    
    if fecha and hora and Po and T and U and Ff and RRR:
        precipitacion = Precipitacion(fecha, hora, Po,T,U,Ff,RRR)
        #insert_one para crear un documento en Mongo
        precipitaciones.insert_one(precipitacion.formato_doc())
        return redirect(url_for('inicioDB'))
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
    return redirect(url_for('inicioDB'))

# Editar o actualizar el contenido 
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

    Po = float(Po)
    T = float(T)
    U= float(U)
    Ff = float(Ff)

    # Utilizaremos la función update_one()
    if fecha and hora and Po and T and U and Ff and RRR:
        precipitaciones.update_one({'fecha': fecha_Precipitacion}, 
                            {'$set': {'fecha' : fecha , 'hora': hora, 'Po': Po,'T': T ,'U': U ,'Ff': Ff ,'RRR': RRR}}) # update_one() necesita de al menos dos parametros para funcionar
        return redirect(url_for('inicioDB'))
    else:
        return "Error de actualización"


# Control del error 404
def error_404(error):
    return render_template('error_404.html'), 404

# Ejecución del programa
if __name__ == '__main__':
    app.register_error_handler(404, error_404)
    app.run(debug = True, port = 2023)