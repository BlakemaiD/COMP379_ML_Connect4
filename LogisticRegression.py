# Logistic Regression for Connect-4
 # Juan Javier Donoso

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA


df = pd.read_csv("connect-4.data", header=None)

# Last column: win/loss/draw
y = df.iloc[:, -1]

# First 42 columns: board positions
X = df.iloc[:, :-1]

from sklearn.preprocessing import OneHotEncoder

# One Hot Encode the board positions
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(X)

# Replace X with the encoded matrix
X = pd.DataFrame(X_encoded)

# Encode labels
label_map = {"win": 1, "loss": -1, "draw": 0}
y = y.replace(label_map)

# 2. TRAIN / VALIDATION / TEST SPLIT (60 / 20 / 20)


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# 3. HYPERPARAMETERS TO TEST


C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
penalties = ["l1", "l2"]

best_model = None
best_acc = 0
best_params = None

results = []

# 4. TRAIN + VALIDATE ALL MODELS


for penalty in penalties:
    for C in C_values:
        try:
            # L1 requires liblinear solver
            solver = "liblinear" if penalty == "l1" else "lbfgs"

            model = LogisticRegression(
                C=C, penalty=penalty, solver=solver, max_iter=500
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            acc = accuracy_score(y_val, preds)
            prec = precision_score(y_val, preds, average=None, zero_division=0)
            rec = recall_score(y_val, preds, average=None, zero_division=0)
            f1 = f1_score(y_val, preds, average=None, zero_division=0)

            results.append((penalty, C, acc))

            # Track best model
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = (penalty, C)

        except Exception as e:
            # Some combinations are invalid (e.g. L1 + lbfgs)
            continue

# 5. PRINT BEST PARAMETERS


print(" BEST LOGISTIC REGRESSION MODEL")
print(f"Penalty: {best_params[0]}")
print(f"C value: {best_params[1]}")
print(f"Validation Accuracy: {best_acc}")

# 6. FINAL TEST SET EVALUATION


test_preds = best_model.predict(X_test)

final_acc = accuracy_score(y_test, test_preds)
final_prec = precision_score(y_test, test_preds, average=None, zero_division=0)
final_rec = recall_score(y_test, test_preds, average=None, zero_division=0)
final_f1 = f1_score(y_test, test_preds, average=None, zero_division=0)

print(" FINAL TEST PERFORMANCE")
print(f"Accuracy: {final_acc}")
print("Class-wise Precision:")
print(f"  draw: {final_prec[0]}")
print(f"  loss: {final_prec[1]}")
print(f"  win : {final_prec[2]}")

print("Class-wise Recall:")
print(f"  draw: {final_rec[0]}")
print(f"  loss: {final_rec[1]}")
print(f"  win : {final_rec[2]}")

print("Class-wise F1 Score:")
print(f"  draw: {final_f1[0]}")
print(f"  loss: {final_f1[1]}")
print(f"  win : {final_f1[2]}")

