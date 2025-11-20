import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
import numpy as np

# fetch dataset 
connect_4 = fetch_ucirepo(id=26)

# data (as pandas dataframes) 
X = connect_4.data.features 
y = connect_4.data.targets 

# Column names for final combined dataframe
cols = [
    'a1','a2','a3','a4','a5','a6',
    'b1','b2','b3','b4','b5','b6',
    'c1','c2','c3','c4','c5','c6',
    'd1','d2','d3','d4','d5','d6',
    'e1','e2','e3','e4','e5','e6',
    'f1','f2','f3','f4','f5','f6',
    'g1','g2','g3','g4','g5','g6',
    'class'
]

# Combine X and y into a single dataframe
df = pd.concat([X, y], axis=1)
df.columns = cols

# Separate features and labels
X = df.drop('class', axis=1)
y = df['class']

# One-hot encode X
oh = OneHotEncoder()
X_enc = oh.fit_transform(X)

# Dataset splits
splits = [
    (0.6, 0.2, 0.2),
    (0.5, 0.25, 0.25),
    (0.2, 0.2, 0.6)
]

# k values to test
k_values = [3, 7, 13]

# Function to compute metrics manually
def compute_metrics(cm):
    # cm layout: rows = true labels, cols = predicted labels
    # Order: ['draw', 'loss', 'win']
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    # Use your formulas:
    accuracy = np.sum(TP) / np.sum(cm)

    precision = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP+FP)!=0)
    recall    = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP+FN)!=0)
    f1        = np.divide(2 * precision * recall, (precision + recall), 
                          out=np.zeros_like(TP, dtype=float), 
                          where=(precision + recall)!=0)

    return accuracy, precision, recall, f1

# Loop through splits
for train_ratio, val_ratio, test_ratio in splits:
    print(f"\n--- Dataset split Train:{train_ratio*100}% Val:{val_ratio*100}% Test:{test_ratio*100}% ---")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_enc, y, test_size=1-train_ratio, stratify=y, random_state=42
    )

    temp_total = val_ratio + test_ratio
    val_size = val_ratio / temp_total

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_size, stratify=y_temp, random_state=42
    )
    
    # Test each k
    for k in k_values:
        print(f"\n### kNN with k={k} ###")

        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Print sklearn classification report
        print(classification_report(y_test, y_pred, digits=3))

        # Compute confusion matrix
        labels = ['draw', 'loss', 'win']
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # Manual metric calculations
        accuracy, precision, recall, f1 = compute_metrics(cm)

        print("\nManual Metrics (Your Formulas):")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClass-wise Precision:")
        for label, p in zip(labels, precision):
            print(f"  {label}: {p:.4f}")
        print("\nClass-wise Recall:")
        for label, r in zip(labels, recall):
            print(f"  {label}: {r:.4f}")
        print("\nClass-wise F1:")
        for label, f in zip(labels, f1):
            print(f"  {label}: {f:.4f}")
