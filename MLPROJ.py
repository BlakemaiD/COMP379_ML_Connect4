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
with open('logreg.pkl', 'rb') as file:
    logreg = pickle.load(file)

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


#8 creating the board
ROWS = 6
COLUMNS = 7

def create_empty_board():
#Board is 6x7, board[row][columns] filled with b to represent a blank board

    board = []
    for i in range(ROWS):
        row = []
        for c in range(COLUMNS):
            row.append('b')
        board.append(row)
    return board

def print_board(board):
    print("--------Current Board--------")
    for r in range(ROWS-1, -1, -1):  # print from top down
        print(' '.join(board[r]))

#will convert the current board to the string for setting_board_string/Sam's code
def board_to_string(board):
    cells = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            cells.append(board[r][c])
    data_string = ','.join(cells)
    return data_string

#when moving you need to check to see if the move is valid in connect 4
def get_moves(board):
    valid_moves = []
    for c in range(COLUMNS):
        column_has_space = False
        for r in range(ROWS):
            if board[r][c] == 'b':
                column_space = True
                valid_moves.append((r, c))
                break   
    return valid_moves

import copy
def generate_boards(board, turn):
    moves = get_moves(board)
    print("Possible moves:", moves)
    possible_boards = []
    for move in moves:
        test_board = copy.deepcopy(board)
        test_board[move[0]][move[1]] = 'x' if turn % 2 == 1 else 'o'
        possible_boards.append(test_board)
    return possible_boards
    
def setting_board_string(data_string):
    #1
    data_array = np.array(data_string.strip().split(','))
    #2
    #unique_categories, indices = np.unique(data_array, return_inverse=True)
    #3.
    N = data_array.size  # should be 42
    C = 3 # 3
    one_hot_encoded = np.zeros((N, C), dtype=int)
    for i, cell in enumerate(data_array):
        if cell == 'b':
            one_hot_encoded[i, 0] = 1  # [1,0,0]
        elif cell == 'o':
            one_hot_encoded[i, 1] = 1  # [0,1,0]
        elif cell == 'x':
            one_hot_encoded[i, 2] = 1  # [0,0,1]

    #4 flatten to (1,126)
    X_flat = one_hot_encoded.flatten()
    X_for_svc = X_flat.reshape(1, -1)

    return X_for_svc
    
def predict(model, X):
    if model == nn1:
        return model.predict_proba(setting_board_string(X))
    if model == nn2:
        return model.predict(setting_board_string(X))
    if model == knn:
        return model.predict_proba(setting_board_string(X))
    if model == svm:
        return model.decision_function(setting_board_string(X))
    if model == logreg:
        return model.decision_function(setting_board_string(X))

def check_win(board, model):

    #check whose turn it is as a parameter, also include a fucntion that will tell when the board is full

    if model == 1:
        cell = 'x'
    else:
        cell = 'o'
        
    #horizontal test
    for r in range(ROWS):
        for c in range(COLUMNS - 3):
            if (board[r][c] == cell
                and board[r][c+1] == cell
                and board[r][c+2] == cell
                and board[r][c+3] == cell):
                return True

    #verical test
    for r in range(ROWS - 3):
        for c in range(COLUMNS):
            if (board[r][c] == cell
                and board[r+1][c] == cell
                and board[r+2][c] == cell
                and board[r+3][c] == cell):
                return True

    #positive diagnal
    for c in range(COLUMNS - 3):
        for r in range(ROWS - 3):
            if (board[r][c] == cell
                and board[r+1][c+1] == cell
                and board[r+2][c+2] == cell
                and board[r+3][c+3] == cell):
                return True

    #negative diagnol
    for r in range(3, ROWS):
        for c in range(COLUMNS - 3):
            if (board[r][c] == cell
                and board[r-1][c+1] == cell
                and board[r-2][c+2] == cell
                and board[r-3][c+3] == cell):
                return True

    return False

def flip_board_string(data_string):
    """
    Swaps 'x' and 'o' in the board string to create a Player 1 (x) perspective.
    This is used when Model 2 (o) is playing.
    """
    # Use string replace to swap symbols
    
    # 1. Replace 'o' with a temporary placeholder (e.g., 'z')
    temp_string = data_string.replace('o', 'z') 
    
    # 2. Replace 'x' with 'o'
    temp_string = temp_string.replace('x', 'o')
    
    # 3. Replace the placeholder 'z' with 'x'
    flipped_string = temp_string.replace('z', 'x')
    
    return flipped_string

def update_wins(model):
    if model == nn1:
        global nn1_wins
        nn1_wins += 1
    if model == nn2:
        global nn2_wins
        nn2_wins += 1
    if model == knn:
        global knn_wins
        knn_wins += 1
    if model == svm:
        global svm_wins
        svm_wins += 1
    if model == logreg:
        global logreg_wins
        logreg_wins += 1
        
      
nn1_wins = 0
nn2_wins = 0
knn_wins = 0
svm_wins = 0
logreg_wins = 0

models = [logreg, nn1, nn2, knn, svm]

for model1 in models:
    for model2 in models:
        turn = 1
        won = -1
        #don't play a model against itself
        if model1 == model2:
            continue
        #clear board before game starts
        board = create_empty_board()
        print("#", model1, "vs", model2, "#")
        print_board(board)
        while won == -1:
            print("----Turn----", turn)
            possible_boards = generate_boards(board, turn)
            print("There are ", len(possible_boards), "possible boards.")
            max_win_prob = float('-inf')
            best_possibility = 0
            count = 0     
            #if its model 1's turn
            if turn % 2 == 1:
                #predict for possible boards
                for possibility in possible_boards:
                    pred = predict(model1, board_to_string(possibility))
                    print("Predictions:", pred)
                    #choose best move
                    if pred[0][2] > max_win_prob:
                        max_win_prob = pred[0][2]
                        best_possibility = count
                    count += 1          
            #if its model 2's turn
            elif turn % 2 == 0:
                #predict for possible boards
                for possibility in possible_boards:
                    board_str_unflipped = board_to_string(possibility)
                    board_str_flipped = flip_board_string(board_str_unflipped)
                    pred = predict(model2, board_str_flipped)
                    print("Predictions:", pred)
                    #choose best move
                    if pred[0][1] > max_win_prob:
                        max_win_prob = pred[0][2]
                        best_possibility = count
                    count += 1
            #update board
            board = copy.deepcopy(possible_boards[best_possibility])
            #print board?
            print_board(board)
            #check win
            if check_win(board, turn % 2):
                won = turn % 2
            if won == 1:
                update_wins(model1)
            if won == 0:
                update_wins(model2)
            #change turns
            turn += 1
        print("NN1 wins:", nn1_wins)
        print("NN2 wins:", nn2_wins)
        print("KNN wins:", knn_wins)
        print("SVM wins:", svm_wins)
        print("Logreg wins:", logreg_wins)













