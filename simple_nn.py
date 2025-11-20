# probably going to rework this to use the same library as sam's

import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("This is the simple neural network Machine Learning Model")
#%%
#Upload the data, once unziped
#/Users/Downloads/connect+4/connect-4.data
data = pd.read_csv("connect-4.data", header=None)

# Lfeatures = first 42 columns
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
#%%
#60/20/20 split, (train / val/ test) need to split the data into various types

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split( #split the last 40 -> 20/20
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Train size:", X_train.shape[0])
print("Val size:", X_val.shape[0])
print("Test size:", X_test.shape[0])
#%%
#One-hot encode X_train and X_val

encoder = OneHotEncoder(handle_unknown='ignore')
X_train_enc = encoder.fit_transform(X_train)
X_val_enc = encoder.transform(X_val)

#%%
#After now doing the last parameters:
#best_score = 0.8143
#best_params = (1, "rbf", "scale")

best_score = 0.0
best_params = None

epochs = 40
batch_size = 128
hidden_layer_size = 2

alpha_values = [0.002, 0.0001, 0.00005]
learning_rates = [0.0001, 0.001, 0.01, 0.1]

# training
for alpha in alpha_values:
    for learning_rate in learning_rates:
        nn = MLPClassifier(activation="identity", batch_size=batch_size, alpha=alpha,
                hidden_layer_sizes=(hidden_layer_size,), random_state=67,
                learning_rate_init=learning_rate, max_iter=epochs)

        nn.fit(X_train_enc, y_train)
        preds = nn.predict(X_val_enc)
        score = accuracy_score(y_val, preds)

        print(f"alpha={alpha}, learning rate={learning_rate}, val acc={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = (alpha, learning_rate)

print(f"\nBest params: {best_params[0]} alpha, {best_params[1]} learning rate")
print("Best validation accuracy:", best_score)

X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

encoder_final = OneHotEncoder(handle_unknown='ignore')
X_train_full_enc = encoder_final.fit_transform(X_train_full)
X_test_enc = encoder_final.transform(X_test)

#%%
# Train final NN using best hyperparameters

alpha_best, learning_rate_best = best_params

final_svm = MLPClassifier(activation="identity", batch_size=batch_size, alpha=alpha,
                    hidden_layer_sizes=(hidden_layer_size,), random_state=67,
                    learning_rate_init=learning_rate_best, max_iter=epochs)

final_svm.fit(X_train_full_enc, y_train_full)

# Predictions
test_preds = final_svm.predict(X_test_enc)
#%%
test_acc = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds, average='weighted')
test_recall = recall_score(y_test, test_preds, average='weighted')
test_f1 = f1_score(y_test, test_preds, average='weighted')

print("Accuracy :", test_acc)
print("Precision:", test_precision)
print("Recall   :", test_recall)
print("F1 Score :", test_f1)
