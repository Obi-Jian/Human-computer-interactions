from ucimlrepo import fetch_ucirepo
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1) Carica dataset
iris = fetch_ucirepo(id=53)
X = iris.data.features        # shape (150, 4) - equivalente di meas
y = iris.data.targets['class'] # shape (150,)  - equivalente di species

print(y.shape)
# 2) Crea il classificatore kNN con k=5 e standardizzazione
# Pipeline applica prima StandardScaler poi kNN
# equivalente di 'Standardize', true in MATLAB
knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# 3) 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(knn, X, y, cv=kf)
print("Scores per fold:", scores)
print(f"Accuratezza media: {scores.mean()*100:.2f}%")
print(f"Misclassification error: {(1 - scores.mean())*100:.2f}%")

# 4) Matrice di confusione
y_pred = cross_val_predict(knn, X, y, cv=kf)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.unique())
disp.plot(cmap='Blues')
plt.title('Matrice di Confusione - kNN (k=5)')
plt.show()