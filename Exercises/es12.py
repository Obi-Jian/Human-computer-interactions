from sklearn import svm
from ucimlrepo import fetch_ucirepo
import numpy as np

# 1) Load dataset
iris = fetch_ucirepo(id=53)

X_full = iris.data.features   # DataFrame con le 4 feature
y_full = iris.data.targets    # DataFrame con la colonna 'class'

# 2) FFilter out Iris Setosa (we only keep Versicolor and Virginica)
mask = y_full['class'].isin(['Iris-versicolor', 'Iris-virginica'])

X_filtered = X_full[mask]
y_filtered = y_full[mask]

# 3) Estrai solo lunghezza e larghezza dei petali (colonne 3 e 4)
X = X_filtered[['petal length', 'petal width']].values   # shape (100, 2)
y = y_filtered['class'].values                            # shape (100,)

print("X shape:", X.shape) # (100, 2) meaning 2 columns with 100 elements each
print("y shape:", y.shape) # (100, ) meaning 1 column with 100 elements. These are the targets we will use to train the svm
print("Classi presenti:", set(y))

# 4) Addestra SVM lineare
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# 5) Estrai i support vectors
sv = clf.support_vectors_
print("Support vectors' shape:", sv)

# 6) Scatter plot con support vectors sovrapposti
import matplotlib.pyplot as plt

# Separa i punti per classe
mask_vers = y == 'Iris-versicolor'
mask_virg = y == 'Iris-virginica'

plt.figure(figsize=(8, 6))

# Scatter delle due classi
plt.scatter(X[mask_vers, 0], X[mask_vers, 1], label='Versicolor', color='blue', alpha=0.6)
plt.scatter(X[mask_virg, 0], X[mask_virg, 1], label='Virginica', color='red', alpha=0.6)

# Sovrapponi i support vectors
plt.scatter(sv[:, 0], sv[:, 1], s=120, facecolors='none', edgecolors='black', linewidths=2, label='Support Vectors')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('SVM - Versicolor vs Virginica')
plt.legend()
plt.show()

# 7) 5 fold cross classification
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

# Configuration
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Performing
# we don't need fit here because the cross_val_score function implements it automatically
scores = cross_val_score(clf, X, y, cv=kf)
print("Cross-validation scores:", scores)
print("Average score:", scores.mean())

# 8) Calculate average accuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Ottieni le predizioni tramite cross-validation
y_pred = cross_val_predict(clf, X, y, cv=kf)

# Genera la matrice di confusione
cm = confusion_matrix(y, y_pred)

# 9) Visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title('Matrice di Confusione (5-Fold CV)')
plt.show()

# The confusion matrix shows how well our svm has classified the dataset after being trained off of it