import keras
import pickle
import sklearn
import pandas as pd
import numpy as np
#load all models
nn2 = keras.models.load_model('nn2.keras')
with open('nn1.pkl', 'rb') as file:
    nn1 = pickle.load(file)
with open('knn.pkl', 'rb') as file:
    knn = pickle.load(file)
with open('svm.pkl', 'rb') as file:
    svm = pickle.load(file)
#test models
data_string = "b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b"
test_array = np.array(data_string.strip().split(','))
print("SVM:", svm.predict(test_array))
print("NN2:", nn2.predict(test_array))
print("NN1:", nn1.predict(test_array))
print("KNN:", knn.predict(test_array))
#data structure for storing game board

#play each model against each other (do one where each goes first?)
    #before starting, clear game board and select two models
    #general loop:
        #determine whose turn it is
        #determine what moves are possible
        #predict chance of winning for each possible moves
        #make the move with highest chance of winning
        #check if the games ends
            #if game ends, record who won (maybe record whether the first player won?)
            #brake loop iteration

        #change whose turn it is




