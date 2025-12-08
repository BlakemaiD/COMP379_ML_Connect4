#%% md
# # This is a sample Jupyter Notebook
# 
# Below is an example of a code cell. 
# Put your cursor into the cell and press Shift+Enter to execute it and select the next one, or click 'Run Cell' button.
# 
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# 
# To learn more about Jupyter Notebooks in PyCharm, see [help](https://www.jetbrains.com/help/pycharm/ipython-notebook-support.html).
# For an overview of PyCharm, go to Help -> Learn IDE features or refer to [our documentation](https://www.jetbrains.com/help/pycharm/getting-started.html).
#%%
print("This is the SVM Machine Learning Model")
print("This is the SVM Machine Learning Model")

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#%%
#Upload the data, once unziped
#/Users/Downloads/connect+4/connect-4.data
data = pd.read_csv("/Users/Downloads/connect+4/connect-4.data", header=None)

# Lfeatures = first 42 columns
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Total samples:", X.shape[0]) #test for concept
print("Num features:", X.shape[1])
#%%
def run_svm_for_split(X, y, train_ratio, val_ratio, test_ratio):
    print(f"\n=== Running SVM for split {int(train_ratio * 100)}/{int(val_ratio * 100)}/{int(test_ratio * 100)} ===")

    # ---- 1. Split data: train vs temp ----
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_ratio),
        random_state=42,
        stratify=y
    )

    # ---- 2. Split temp into val vs test ----
    # temp = val + test, so we adjust ratio relative to that
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_fraction_of_temp),
        random_state=42,
        stratify=y_temp
    )

    print("Train size:", X_train.shape[0])
    print("Val size  :", X_val.shape[0])
    print("Test size :", X_test.shape[0])

    # ---- 3. One-hot encode train + val ----
    encoder = OneHotEncoder(handle_unknown='ignore')

    X_train_enc = encoder.fit_transform(X_train)
    X_val_enc = encoder.transform(X_val)

    # ---- 4. Grid search over C and kernel (no gamma tuning, gamma='scale') ----
    best_score = 0.0
    best_params = None

    C_values = [0.1, 1]
    kernels = ["linear", "rbf"]

    for C in C_values:
        for kernel in kernels:
            svm = SVC(C=C, kernel=kernel, gamma="scale")

            svm.fit(X_train_enc, y_train)
            preds = svm.predict(X_val_enc)
            score = accuracy_score(y_val, preds)

            print(f"C={C}, kernel={kernel}, val acc={score:.4f}")

            if score > best_score:
                best_score = score
                best_params = (C, kernel)

    print("\nBest params:", best_params)
    print("Best validation accuracy:", best_score)

    # ---- 5. Retrain on TRAIN + VAL with best hyperparameters ----
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    encoder_final = OneHotEncoder(handle_unknown='ignore')
    X_train_full_enc = encoder_final.fit_transform(X_train_full)
    X_test_enc = encoder_final.transform(X_test)

    C_best, kernel_best = best_params

    final_svm = SVC(
        C=C_best,
        kernel=kernel_best,
        gamma="scale"
    )

    final_svm.fit(X_train_full_enc, y_train_full)

    # ---- 6. Test metrics ----
    test_preds = final_svm.predict(X_test_enc)

    test_acc = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, average='weighted')
    test_recall = recall_score(y_test, test_preds, average='weighted')
    test_f1 = f1_score(y_test, test_preds, average='weighted')

    print("\n=== FINAL TEST METRICS ===")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1 Score : {test_f1:.4f}")

    # ---- 7. Return a dict you can use for tables/summary ----
    return {
        "split": (train_ratio, val_ratio, test_ratio),
        "best_params": best_params,
        "val_acc": best_score,
        "test_acc": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1
    }

# 60 / 20 / 20
result_60_20_20 = run_svm_for_split(X, y, 0.60, 0.20, 0.20)

# 50 / 25 / 25
result_50_25_25 = run_svm_for_split(X, y, 0.50, 0.25, 0.25)

# 20 / 20 / 60
result_20_20_60 = run_svm_for_split(X, y, 0.20, 0.20, 0.60)

#%% md
# 
# === Running SVM for split 60/20/20 ===
# Train size: 40534
# Val size  : 13511
# Test size : 13512
# C=0.1, kernel=linear, val acc=0.7558
# C=0.1, kernel=rbf, val acc=0.7664
# C=1, kernel=linear, val acc=0.7563
# C=1, kernel=rbf, val acc=0.8147
# 
# Best params: (1, 'rbf')
# Best validation accuracy: 0.814669528532307
# 
# === FINAL TEST METRICS ===
# Accuracy : 0.8179
# Precision: 0.7986
# Recall   : 0.8179
# F1 Score : 0.7856
# 
# === Running SVM for split 50/25/25 ===
# Train size: 33778
# Val size  : 16889
# Test size : 16890
# C=0.1, kernel=linear, val acc=0.7560
# C=0.1, kernel=rbf, val acc=0.7602
# C=1, kernel=linear, val acc=0.7562
# C=1, kernel=rbf, val acc=0.8097
# 
# Best params: (1, 'rbf')
# Best validation accuracy: 0.8096986204038131
# 
# === FINAL TEST METRICS ===
# Accuracy : 0.8189
# Precision: 0.7939
# Recall   : 0.8189
# F1 Score : 0.7852
# 
# === Running SVM for split 20/20/60 ===
# Train size: 13511
# Val size  : 13511
# Test size : 40535
# C=0.1, kernel=linear, val acc=0.7535
# C=0.1, kernel=rbf, val acc=0.7310
# C=1, kernel=linear, val acc=0.7546
# C=1, kernel=rbf, val acc=0.7849
# 
# Best params: (1, 'rbf')
# Best validation accuracy: 0.7849159943749537
# 
# === FINAL TEST METRICS ===
# Accuracy : 0.8062
# Precision: 0.7837
# Recall   : 0.8062
# F1 Score : 0.7693
# 
#%%

