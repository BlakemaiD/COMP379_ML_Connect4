import keras
import pickle
#load all models
nn2 = keras.models.load_model('nn2.keras')
with open('nn1.pkl', 'rb') as file:
    nn1 = pickle.load(file)

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

