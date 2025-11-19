
print("This is the SVM Machine Learning Model")

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#%%
#Upload the data, once unziped
#/Users/Downloads/connect+4/connect-4.data
data = pd.read_csv("/Users/Downloads/connect+4/connect-4.data", header=None)

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


#C_values = [0.1, 1, 10] -> 10 took too long
#kernels = ["linear", "rbf"]
#gammas = ["scale", "auto"]

C_values = [0.1, 1]
kernels = ["linear", "rbf"]
gammas = ["scale", "auto"]



for C in C_values:
    for kernel in kernels:
        for gamma in gammas:

            svm = SVC(C=C, kernel=kernel, gamma=gamma)

            svm.fit(X_train_enc, y_train)
            preds = svm.predict(X_val_enc)
            score = accuracy_score(y_val, preds)

            print(f"C={C}, kernel={kernel}, val acc={score:.4f}")
            if score > best_score:
                best_score = score
                best_params = (C, kernel, gamma)

print("\nBest params:", best_params)
print("Best validation accuracy:", best_score)

#%% md
# Output for 60/20/20
# 
# C=0.1, kernel=linear, val acc=0.7558
# C=0.1, kernel=rbf, val acc=0.7663
# C=1, kernel=linear, val acc=0.7562
# C=1, kernel=rbf, val acc=0.8143
# 
# Best params: (1, 'RBF Kernel)
# Best validation accuracy: 0.8143
#%%
#fit encoder on train and val data

X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

encoder_final = OneHotEncoder(handle_unknown='ignore')
X_train_full_enc = encoder_final.fit_transform(X_train_full)
X_test_enc = encoder_final.transform(X_test)

#%%
# Train final SVM using best hyperparameters

C_best, kernel_best, gamma_best = best_params

final_svm = SVC(
    C=C_best,
    kernel=kernel_best,
    gamma=gamma_best
)

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