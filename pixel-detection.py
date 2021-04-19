Deep Neural Network con Keras
En este ejercicio se procede a realizar una Deep Neural Network con el Framework Keras, utilizando Tensorflow como Backend.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout
from keras.models import Model
from sklearn.metrics import roc_auc_score
from keras.layers import Wrapper
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras import regularizers
import matplotlib.pyplot as plt
# Feature Scaling
from sklearn.preprocessing import StandardScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Using TensorFlow backend.
['sample_submission.csv', 'test.csv', 'train.csv']
0. Funciones auxiliares.
precisiones_globales=[]
epochs = 15
def graf_model(train_history):
    f = plt.figure(figsize=(15,10))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    # summarize history for accuracy
    ax.plot(train_history.history['binary_accuracy'])
    ax.plot(train_history.history['val_binary_accuracy'])
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    ax2.plot(train_history.history['loss'])
    ax2.plot(train_history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')
    plt.show()
def precision(model, registrar=False):
    y_pred = model.predict(train_dfX)
    train_auc = roc_auc_score(train_dfY, y_pred)
    y_pred = model.predict(val_dfX)
    val_auc = roc_auc_score(val_dfY, y_pred)
    print('Train AUC: ', train_auc)
    print('Vali AUC: ', val_auc)
    if registrar:
        precisiones_globales.append([train_auc,val_auc])
1. Importando los datos:
Se procede a leer tanto el train.csv que es el conjunto de entrenamiento como el test.csv que es el conjunto de validacion. Mas adelante se construira en base a train.csv el conjunto de dev

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
Train shape :  (200000, 202)
Test shape :  (200000, 201)
Se crea las variables X y Y con las que se van a entrenar al modelo. Tambien se elimina el ID_Code ya que no aporta valor al entrenamiento del modelo.

train_dfX = train_df.drop(['ID_code', 'target'], axis=1)
train_dfY = train_df['target']
submission = test_df[['ID_code']].copy()
test_df = test_df.drop(['ID_code'], axis=1)
2. Normalizando los valores de X
Se transforma X de manera que quede normalizado todos su valores.

sc = StandardScaler()
train_dfX = sc.fit_transform(train_dfX)
test_df = sc.transform(test_df)
3. Separando Entrenamiento de Validacion
Se divide X para tener un conjunto de entrenamiento y otro de validacion y pruebas. El 90% de las observaciones quedan en el conjunto de entrenamiento.

train_dfX,val_dfX,train_dfY, val_dfY = train_test_split(train_dfX,train_dfY , test_size=0.1, stratify=train_dfY)
print("Entrnamiento: ",train_dfX.shape)
print("Validacion : ",val_dfX.shape)
Entrnamiento:  (180000, 200)
Validacion :  (20000, 200)
Aqui podemos ver como nuestra variable de entramiento train_dfX tiene 16200 observaciones y 200 caracteristicas. Es decir, la forma en la que ingresaremos datos a nuestro modelo sera (200, )

4. Creacion del modelo.
Vamos a definir una funcion que cree un modelo de red neuronal. Esta red va a tener 2 capaz ocultas con 1028 neuronas cada una. Se va a utilizar el optimizador SGD y la funcion de costo BinaryCrossEntropy. Los valores W del modelo se iniciaran de manera aleatoria y uniforme, los valores de b se iniciaran en cero.

A tener en cuenta:

Cual es el learning Rate de este modelo?

def func_model():   
    inp = Input(shape=(200,))
    x=Dense(1028, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(inp)
    x=Dense(1028, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(x) 
    x=Dense(1, activation="sigmoid", kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['binary_accuracy'])
    return model
model = func_model()
print(model.summary())
WARNING:tensorflow:From /opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 200)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1028)              206628    
_________________________________________________________________
dense_2 (Dense)              (None, 1028)              1057812   
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 1029      
=================================================================
Total params: 1,265,469
Trainable params: 1,265,469
Non-trainable params: 0
_________________________________________________________________
None
5. Entrenamiento del modelo.
Se entrenara el modelo con un batch_size de 512. Utilizando nuestras variables de validacion para seguir de cerca el accuracy tanto en entrenamiento como en validacion. Esto lo haremos por 20 epochs.

train_history = model.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))
WARNING:tensorflow:From /opt/conda/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Train on 180000 samples, validate on 20000 samples
Epoch 1/15
180000/180000 [==============================] - 3s 14us/step - loss: 0.3354 - binary_accuracy: 0.8991 - val_loss: 0.3170 - val_binary_accuracy: 0.8995
Epoch 2/15
180000/180000 [==============================] - 1s 7us/step - loss: 0.3089 - binary_accuracy: 0.8995 - val_loss: 0.3002 - val_binary_accuracy: 0.8995
Epoch 3/15
180000/180000 [==============================] - 1s 7us/step - loss: 0.2916 - binary_accuracy: 0.8995 - val_loss: 0.2827 - val_binary_accuracy: 0.8995
Epoch 4/15
180000/180000 [==============================] - 1s 7us/step - loss: 0.2741 - binary_accuracy: 0.8996 - val_loss: 0.2659 - val_binary_accuracy: 0.9001
Epoch 5/15
180000/180000 [==============================] - 1s 8us/step - loss: 0.2589 - binary_accuracy: 0.9020 - val_loss: 0.2531 - val_binary_accuracy: 0.9035
Epoch 6/15
180000/180000 [==============================] - 1s 8us/step - loss: 0.2485 - binary_accuracy: 0.9066 - val_loss: 0.2456 - val_binary_accuracy: 0.9073
Epoch 7/15
180000/180000 [==============================] - 1s 8us/step - loss: 0.2428 - binary_accuracy: 0.9098 - val_loss: 0.2421 - val_binary_accuracy: 0.9083
Epoch 8/15
180000/180000 [==============================] - 1s 8us/step - loss: 0.2400 - binary_accuracy: 0.9114 - val_loss: 0.2405 - val_binary_accuracy: 0.9099
Epoch 9/15
180000/180000 [==============================] - 1s 8us/step - loss: 0.2386 - binary_accuracy: 0.9120 - val_loss: 0.2398 - val_binary_accuracy: 0.9101
Epoch 10/15
180000/180000 [==============================] - 1s 8us/step - loss: 0.2377 - binary_accuracy: 0.9123 - val_loss: 0.2395 - val_binary_accuracy: 0.9107
Epoch 11/15
180000/180000 [==============================] - 1s 8us/step - loss: 0.2371 - binary_accuracy: 0.9125 - val_loss: 0.2393 - val_binary_accuracy: 0.9108
Epoch 12/15
180000/180000 [==============================] - 1s 8us/step - loss: 0.2365 - binary_accuracy: 0.9128 - val_loss: 0.2392 - val_binary_accuracy: 0.9111
Epoch 13/15
180000/180000 [==============================] - 1s 8us/step - loss: 0.2360 - binary_accuracy: 0.9128 - val_loss: 0.2390 - val_binary_accuracy: 0.9111
Epoch 14/15
180000/180000 [==============================] - 1s 8us/step - loss: 0.2355 - binary_accuracy: 0.9130 - val_loss: 0.2390 - val_binary_accuracy: 0.9107
Epoch 15/15
180000/180000 [==============================] - 1s 7us/step - loss: 0.2351 - binary_accuracy: 0.9132 - val_loss: 0.2388 - val_binary_accuracy: 0.9111
Vamos a ver una grafica de como se comporta nuestro modelo tanto en la perdida como en la precision (binary_accuracy).

graf_model(train_history)

6. Precision del modelo.
En esta competencia la metrica utilizada para medir la precision de nuestro modelo es el AUC (referencia rapida: https://es.wikipedia.org/wiki/Curva_ROC). Por lo que vamos a estar bastante enfocados en mejorarla.

La funcion precision() nos va a permitir, dado como parametro uno de nuestros modelos, imprimir el valor de AUC tanto del conjunto de entrenamiento como del de validacion.

precision(model, True)
Train AUC:  0.856625412544387
Vali AUC:  0.8530720770798592
7. Tamaño de la Red. (Evaluado)
En este apartado vamos a estar revisando como el tamaño de la red afecta el desempeño de la misma. Vamos a experimentar utilizando distinta cantidad de capaz ocultas y de neuronas por capa.

La variable arquitectura es una lista de Python que sirve como parametro de la funcion model(arquitectura) para definir una red neuronal con una cantidad len(arquitectura) de capaz ocultas, donde cada capa oculta l tiene una cantidad de neuronas arquitectura[l-1].

Por ejemplo, de configurar la variable arquitectura de forma [5,10,20] tendremos una red neuronal con 3 capaz ocultas, con 5, 10 y 20 capaz respectivamente.

def func_model(arquitectura): 
    first =True
    inp = Input(shape=(200,))
    for capa in arquitectura:        
        if first:
            x=Dense(capa, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(inp)            
            first = False
        else:
            x=Dense(capa, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(x)  
    x=Dense(1, activation="sigmoid", kernel_initializer='random_uniform', bias_initializer='zeros')(x)  
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['binary_accuracy'])
    return model
Experimento 1
Juega con distintos tamaños de red para ver como esto afecta el desempeño del modelo (Vas a poder ver el impacto en las graficas de perdida, binary_accuracy y valor de AUC)

Vas a poder comparar valores utilizando el Experimento 1 y 2 al mismo tiempo.

arquitectura1 = None
model1 = func_model(arquitectura1)
#Para revisar la estructura del modelo, quitar el comentario de la instruccion siguiente:
#print(model1.summary())
train_history_tam1 = model1.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY), verbose=0)
graf_model(train_history_tam1)
precision(model1)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-12-5694544ca2e3> in <module>()
      1 arquitectura1 = None
----> 2 model1 = func_model(arquitectura1)
      3 #Para revisar la estructura del modelo, quitar el comentario de la instruccion siguiente:
      4 #print(model1.summary())
      5 train_history_tam1 = model1.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY), verbose=0)

<ipython-input-11-7b25f8fc5fd6> in func_model(arquitectura)
      2     first =True
      3     inp = Input(shape=(200,))
----> 4     for capa in arquitectura:
      5         if first:
      6             x=Dense(capa, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(inp)

TypeError: 'NoneType' object is not iterable
Experimento 2
arquitectura2 = None
model2 = func_model(arquitectura2)
#print(model2.summary())
train_history_tam2 = model2.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))
graf_model(train_history_tam2)
precision(model2)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-13-e785af57da2d> in <module>()
      1 arquitectura2 = None
----> 2 model2 = func_model(arquitectura2)
      3 #print(model2.summary())
      4 train_history_tam2 = model2.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))
      5 graf_model(train_history_tam2)

<ipython-input-11-7b25f8fc5fd6> in func_model(arquitectura)
      2     first =True
      3     inp = Input(shape=(200,))
----> 4     for capa in arquitectura:
      5         if first:
      6             x=Dense(capa, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(inp)

TypeError: 'NoneType' object is not iterable
Resultado Final
Elije una arquitectura de red que te permita tener el valor de AUC mas alto que puedas lograr

Ejecuta esta celda 1 sola vez con la arquitectura final elegida en los experimentos anteriores

arquitecturaFinal = None
modelF = func_model(arquitecturaFinal)
print(modelF.summary())
train_history_tamF = modelF.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))
graf_model(train_history_tamF)
precision(modelF, True)
assert(len(precisiones_globales)==2)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-14-7511fe1b0e40> in <module>()
      1 arquitecturaFinal = None
----> 2 modelF = func_model(arquitecturaFinal)
      3 print(modelF.summary())
      4 train_history_tamF = modelF.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))
      5 graf_model(train_history_tamF)

<ipython-input-11-7b25f8fc5fd6> in func_model(arquitectura)
      2     first =True
      3     inp = Input(shape=(200,))
----> 4     for capa in arquitectura:
      5         if first:
      6             x=Dense(capa, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')(inp)

TypeError: 'NoneType' object is not iterable
7. Regularizacion de la Red. (Evaluado)
En el siguiente ejercicio vamos a experimentar con la regularizacion de nuestra red. La idea es que basado en la arquitectura "optima" que consiguieron en el apartado anterior empecemos a mejorar nuestro modelo.

Vamos a experimentar con distintos valores para el P del DropOut y aplicando o no regularizacion L2 en las distintas capas. Para agregar regularizacion L2 a una capa se debe colocar regularizers.l2(0.01) como parametro en kernel_regularizer.

def func_model_reg():   
    inp = Input(shape=(200,))
    x=Dropout(0)(inp)
    x=Dense(1028, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)
    x=Dropout(0)(x)
    x=Dense(1028, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=None)(x)
    x=Dropout(0)(x)
    x=Dense(1028, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=None)(x)
    x=Dropout(0)(x)  
    x=Dense(1, activation="sigmoid", kernel_initializer='random_uniform', bias_initializer='zeros')(x) 
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['binary_accuracy'])
    return model
Experimento
Prueba modificando la funcion de arriba para ver que resultados da sobre la precision.

model1 = func_model_reg()
#Para revisar la estructura del modelo, quitar el comentario de la instruccion siguiente:
#print(model1.summary())
train_history_tam1 = model1.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY), verbose=0)
graf_model(train_history_tam1)
precision(model1)

Train AUC:  0.8632180151141231
Vali AUC:  0.8589391287033427
Resultado Final
Elije una arquitectura de red que te permita tener el valor de AUC mas alto que puedas lograr

Ejecuta esta celda 1 sola vez con la arquitectura final elegida en los experimentos anteriores

modelRF = func_model_reg()
print(modelRF.summary())
train_history_regF = modelRF.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))
graf_model(train_history_regF)
precision(modelRF, True)
assert(len(precisiones_globales)==3)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_6 (InputLayer)         (None, 200)               0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 200)               0         
_________________________________________________________________
dense_8 (Dense)              (None, 1028)              206628    
_________________________________________________________________
dropout_6 (Dropout)          (None, 1028)              0         
_________________________________________________________________
dense_9 (Dense)              (None, 1028)              1057812   
_________________________________________________________________
dropout_7 (Dropout)          (None, 1028)              0         
_________________________________________________________________
dense_10 (Dense)             (None, 1028)              1057812   
_________________________________________________________________
dropout_8 (Dropout)          (None, 1028)              0         
_________________________________________________________________
dense_11 (Dense)             (None, 1)                 1029      
=================================================================
Total params: 2,323,281
Trainable params: 2,323,281
Non-trainable params: 0
_________________________________________________________________
None
Train on 180000 samples, validate on 20000 samples
Epoch 1/15
180000/180000 [==============================] - 2s 12us/step - loss: 1.9460 - binary_accuracy: 0.8988 - val_loss: 1.8161 - val_binary_accuracy: 0.8995
Epoch 2/15
180000/180000 [==============================] - 2s 10us/step - loss: 1.7103 - binary_accuracy: 0.8995 - val_loss: 1.6084 - val_binary_accuracy: 0.8995
Epoch 3/15
180000/180000 [==============================] - 2s 10us/step - loss: 1.5151 - binary_accuracy: 0.8995 - val_loss: 1.4250 - val_binary_accuracy: 0.8995
Epoch 4/15
180000/180000 [==============================] - 2s 10us/step - loss: 1.3419 - binary_accuracy: 0.8995 - val_loss: 1.2617 - val_binary_accuracy: 0.8995
Epoch 5/15
180000/180000 [==============================] - 2s 9us/step - loss: 1.1881 - binary_accuracy: 0.8995 - val_loss: 1.1175 - val_binary_accuracy: 0.8996
Epoch 6/15
180000/180000 [==============================] - 2s 9us/step - loss: 1.0542 - binary_accuracy: 0.9008 - val_loss: 0.9941 - val_binary_accuracy: 0.9017
Epoch 7/15
180000/180000 [==============================] - 2s 9us/step - loss: 0.9410 - binary_accuracy: 0.9054 - val_loss: 0.8907 - val_binary_accuracy: 0.9065
Epoch 8/15
180000/180000 [==============================] - 2s 9us/step - loss: 0.8459 - binary_accuracy: 0.9092 - val_loss: 0.8035 - val_binary_accuracy: 0.9094
Epoch 9/15
180000/180000 [==============================] - 2s 9us/step - loss: 0.7648 - binary_accuracy: 0.9113 - val_loss: 0.7288 - val_binary_accuracy: 0.9103
Epoch 10/15
180000/180000 [==============================] - 2s 9us/step - loss: 0.6949 - binary_accuracy: 0.9123 - val_loss: 0.6642 - val_binary_accuracy: 0.9102
Epoch 11/15
180000/180000 [==============================] - 2s 9us/step - loss: 0.6342 - binary_accuracy: 0.9126 - val_loss: 0.6082 - val_binary_accuracy: 0.9104
Epoch 12/15
180000/180000 [==============================] - 2s 9us/step - loss: 0.5815 - binary_accuracy: 0.9132 - val_loss: 0.5595 - val_binary_accuracy: 0.9105
Epoch 13/15
180000/180000 [==============================] - 2s 9us/step - loss: 0.5355 - binary_accuracy: 0.9136 - val_loss: 0.5171 - val_binary_accuracy: 0.9108
Epoch 14/15
180000/180000 [==============================] - 2s 9us/step - loss: 0.4955 - binary_accuracy: 0.9139 - val_loss: 0.4803 - val_binary_accuracy: 0.9109
Epoch 15/15
180000/180000 [==============================] - 2s 9us/step - loss: 0.4606 - binary_accuracy: 0.9142 - val_loss: 0.4484 - val_binary_accuracy: 0.9106

Train AUC:  0.8636936128195092
Vali AUC:  0.8590799753317901
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
<ipython-input-17-0915a45dda66> in <module>()
      4 graf_model(train_history_regF)
      5 precision(modelRF, True)
----> 6 assert(len(precisiones_globales)==3)

AssertionError: 
Modelo Final.
Juega con el modelo y genera el mejor valor de AUC que puedas lograr.

Salida de datos.
Con esta celda se exportaran los datos.

y_test = model.predict(test_df)
submission['target'] = y_test
submission.to_csv('submission.csv', index=False)
