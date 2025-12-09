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
#test models
# 1. Input Data String
data_string = "b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b"

# 2. Convert the string data into a 1D NumPy array
# .strip() removes any leading/trailing whitespace, and .split(',') separates the elements
data_array = np.array(data_string.strip().split(','))

# Verify the initial array
print("--- 1. Original 1D NumPy Array ({} elements) ---".format(data_array.size))
print(data_array)
print("-" * 50)


# 3. Identify unique categories and create a mapping for the one-hot encoding
# The unique function also returns the indices of the original array that map to the unique elements
unique_categories, indices = np.unique(data_array, return_inverse=True)

# N = Number of samples (42)
N = data_array.size
# C = Number of categories (3: 'b', 'o', 'x')
C = unique_categories.size

print("Unique Categories (Index Order): {}".format(unique_categories))
# The mapping: b -> 0, o -> 1, x -> 2 (since np.unique sorts the categories alphabetically)
print("-" * 50)


# 4. Create the One-Hot Encoded Array
# Initialize a new array of zeros with shape (N, C) -> (42, 3)
one_hot_encoded = np.zeros((N, C), dtype=int)

# Use the 'indices' (from return_inverse=True) to map elements to the correct column.
one_hot_encoded[np.arange(N), indices] = 1

# 5. Display the intermediate result
print("--- 2. Intermediate One-Hot Encoded (42 samples, 3 features per sample) ---")
print(f"Shape: {one_hot_encoded.shape}")
print(one_hot_encoded[:5])
print("... (showing first 5 rows)")
print("-" * 50)

# 6. CRUCIAL TRANSFORMATION: Flatten and Reshape for the SVC Model
# The SVC model expects ONE sample with 126 features (42 * 3).
# Flatten the (42, 3) matrix into a 1D vector of length 126
X_flat = one_hot_encoded.flatten()

# Reshape the 1D vector into a 2D array with 1 row and 126 columns
# This is the standard (n_samples, n_features) format for a single prediction.
X_for_svc = X_flat.reshape(1, -1)

# 7. Display the final result for the classifier
print("--- 3. Final Feature Vector for SVC Prediction ---")
print(f"Required Shape: {X_for_svc.shape}")
# The new shape (1, 126) matches the 126 features the SVC expects.
print(X_for_svc)
print("-" * 50)

# Reference for the original category mapping
print("Category Mapping:")
print(f"Index 0: {unique_categories[0]}")
print(f"Index 1: {unique_categories[1]}")
print(f"Index 2: {unique_categories[2]}")
print("-" * 50)

#creating a fucntion that combines most of the code above***
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


print("SVM:", svm.decision_function(X_for_svc))
print("NN2:", nn2.predict(X_for_svc))
print("NN1:", nn1.predict_proba(X_for_svc))
print("KNN:", knn.predict_proba(X_for_svc))
print("Logreg:", logreg.predict_proba(X_for_svc))
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

def flip_board(board):
    new_board = []
    for r in range(ROWS):
        row = []
        for c in range(COLUMNS):
            if board[r][c] == 'x':
                row.append('o')
            elif board[r][c] == 'o':
                row.append('x')
            else:
                row.append('b')
        new_board.append(row)
    return new_board


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
                break
            if column_has_space:
                valid_moves.append((c))
    return valid_moves

board = create_empty_board()
print_board(board)
print("Valid moves on empty board:", get_moves(board))


def check_win(board, cell):

    #check whose turn it is as a parameter, also include a fucntion that will tell when the board is full

    #if model == 1:  cell = 'x'
    ##else: cell = 'o'

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

def check_draw(board):

    for r in range(ROWS):
        for c in range(COLUMNS):
            if (board[r][c] == 'b'):
                return False
    return True










