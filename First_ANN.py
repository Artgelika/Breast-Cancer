# -*- coding: utf-8 -*-
"""
Created - July 2020
@author: Angelika

Uniwersalna siec, ktora w zamysle bedzie pasowac do wielu danych, ktore uprzednio nalezy
potraktowac skryptem, aby dane mialy odpowiednia forme: [..,..,..][0,1].
Jest to polaczenie sieci z Kaggle oraz mojej inwencji tworczej.
"""
# importing libraries
import numpy as np
# import pandas as pd
# import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
import tensorflow as tf
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split #do podzialu na train i test
from sklearn.preprocessing import StandardScaler #standaryzowanie danych - feature scaling
# from sklearn.metrics import confusion_matrix # - do tworzenia confusion matrix
from copy import copy



# =============================================================================
# Wczytywanie danych
# =============================================================================

# input_data to lista list (przykładowy kawałek: [['0', '1'], ['0', '1'],..])
# tutaj analizujemy drugą kolumnę z danych, tą, która daje informacje TAK - NIE

def wartosci(input_data:[]): 
    for i in range(len(input_data)): # wsrod informacji (linii) ze wszystkich danych, czy jest zdrowy czy chory
        temp = []
        fragmenty = input_data[i].split(',') # fragmenty to np.: ['1', '0'], czyli pojedynczy element z input_data
                                             # rozłącza elementy rodzielone przecinkami -> [..,..], 
        for j in range(len(fragmenty)): # czyli 0 albo 1
            try:
                temp.append(float(fragmenty[j])) # dodaje to do tymczasowej listy
            except:
                print("Wartość: ",fragmenty[j],"Nie jest liczbą! Linia -> ", i,"Pozycja ->", j)
                exit(1)
        input_data[i] = fragmenty # zamiana listy stringow ['1', '0'] na inty 1,0 (to drugie to input_data)
    return input_data


def load(nazwa:str): 
    try:
        f = open("input\\" + nazwa + ".csv","r")
        data = f.readlines()
        f.close()
    except:
        print("Nothing to present")
        exit(1) # means there was some issue/error/problem and that is why the program is exiting
                # exit(0) means a clean exit without any errors/problems
    input_data = []
    output_data = []

    for i in range(len(data)):
        linia = data[i]
        if (linia == ""): # mija puste linie
            continue

        macierze = linia.split("[") # rozdziela dane, gdy widzi znak "["
        #IN---
        macierz_in = macierze[1].strip() # strip - usuwanie białych znakow
                                         # linia danych z "]" jako ostatni znak danego elementu 
        macierz_in = macierz_in[:len(macierz_in)-1] # macierz in - usunieto "]", aby byly same liczby
        
        #OUT---
        macierz_out = macierze[2].strip() # linia danych z "]" jako ostatni znak danego elementu 
        macierz_out = macierz_out[:len(macierz_out)-1] # macierz out - usunieto "]", aby byly same liczby

        input_data.append(macierz_in)
        output_data.append(macierz_out)

    input_data = wartosci(input_data) # to jest lista list z wartosciami (cale linie z danymi)
    output_data = wartosci(output_data) # to jest lista list ["0", "1"] czyli TAK-NIE
    
    input_data = np.array(input_data).astype(float) # zamiana na tablice numpy
    output_data = np.array(output_data).astype(float)
    
    return input_data, output_data


# =============================================================================
# Normalizacja danych
# =============================================================================

def normalize_data(out):
    maks = out[0][0]
    temp=[]
    for i in out:
        for j in i:
            temp.append(j)    
    maks = max(temp)
    print("Max:",maks)

    out /= maks + 1.0

    return out, maks

# =============================================================================
# Ziarno losowosci
# =============================================================================
def set_seed(Seed:int):
    from random import seed as s
    s(Seed)

    from numpy.random import seed
    seed(Seed)

    from tensorflow import random
    random.set_seed(Seed)

    print("SEED set to:" , Seed)
    
    
# =============================================================================
# Wywołanie konkretnych danych, zmienne początkowe. __MAIN__.
# =============================================================================
if __name__ == "__main__":
    print("Nowa sieć neuronowa")  
    
    #Zmienne początkowe
    SEED = 4
    CPU = False 
    LEARNING_RATE = 0.00025
    BATCH_SIZE = 2000
    Iterations = 5000
    FILE_NAME = "data"
    
    
    # =============================================================================
    # Konfigurowanie sieci
    # =============================================================================
    #setting SEED
    set_seed(SEED)

  
    if (CPU):
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(16)
    else:
        tf.config.experimental.list_physical_devices('GPU')

    
    #wczytywanie danych
    input_data, output_data = load(FILE_NAME)
    
    #normalizacja danych (treningowych (?))
    input_data, maks = normalize_data(input_data)
    #input_data bedzie się zawieralo w wąskim przedziale, aby ułatwić naukę sieci

    output_data2 = []  # tu beda pojedyncze wartosci
    for i in output_data: 
        output_data2.append(int(i[0]))
    labelencoder_X_1 = LabelEncoder() #'e)'
    output_data = labelencoder_X_1.fit_transform(output_data2) #'e)'
    
    #Żeby użyc powyższych funkcji, należy jako output_data dać listę 1D, a tu jest wieksza
    
    # =============================================================================
    # Podzial na czesc treningową i testową
    # =============================================================================
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size = 0.1, random_state = SEED) #X, y; random_state = 0
    print("X_train", X_train)
    print("X_test", X_test)
    print("y_train", y_train)
    print("y_test", y_test)
    
    # feature scaling -> using from sklearn.preprocessing import StandardScaler 
    # is calculated as: z = (x - u) / s; u - srednia probek, s-odchylenie standardowe
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # bez tego: jest jedna tablica tablic [[1., 0.],[1., 0.],..]
    # z tym: tablica array tablic podwojnych array([[[0.,1.], [1., 0.]],
    # ...,[[0.,1.], [1., 0.]]], dtype=float32)
    y_train = to_categorical(y_train) 
    y_test = to_categorical(y_test)
    
    # =============================================================================
    # Model sieci
    # =============================================================================
    #inicjalizowanie sieci ANN
    '''
    input_dim - number of columns of the datasets
    output_dim (units or default units) - number of outputs to be fed to the next layer, if any
    activation - activation function which is ReLU in this case
    init - the way in which weights should be provided to an ANN
    '''
    
    model = Sequential([
        Dense(16, activation = 'relu', input_dim=len(X_train[0])),
        Dropout(0.1),
        Dense(1, activation = 'sigmoid')
        ])
    ''' output_dim (units) is 1 as we want only 1 output from the final layer'''
    
    
    
    # compiling the ANN
    model.compile(Adam(lr=LEARNING_RATE), loss = 'mean_squared_error', metrics=['accuracy'])
    
# =============================================================================
# Fitting the ANN to the Training set 
# =============================================================================
    # y_train powinno byc 1D
    y_train2 = []
    for el in y_train:
        y_train2.append(el[0])
    y_train2 = np.array(y_train2) # changing into numpy array
    
    # model.fit(X_train, y_train, epochs = Iterations, batch_size = BATCH_SIZE) #set to epochs = 150!!
    train = model.fit(copy(X_train), copy(y_train2), batch_size=BATCH_SIZE, epochs=Iterations, verbose=1)
    # verbose=1 will show you an animated progress bar like this: [=============]

    #predicting the test set results
    # y_pred = model.predict(X_test)
    # y_pred = (y_pred > 0.5)
    
    # # #making the Confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    
    # print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
