""" 
Il nostro obiettivo è quello di creare un albero di decisione su un dataset con queste caratteristiche:
• sia binario, univariato e con divisioni a intervallo
• contenga in ogni foglia un numero minimo m di esempi (m lo scegliamo arbritariamente)
• abbia un tasso di errore di classificazione minimo (errore = campioni classificati male / campioni totali)

Nota:
Binario: con solo due divisioni: dx o sx
Con divisioni a intervallo: ossia divisioni che comprendano range di valori, non caratteristiche precisi. Es una divisione con lunghezza > x. Es sbagliato può essere una divisione per colore rosso, perchè non è un range

Come troviamo questi split?
Essendo un problema NP-hard, possiamo solo usare algoritmi euristici, con l'obiettivo di trovare l'indice di impurità

In questo useremo l'algoritmo CART:
1. prendere una variabile;
2. generare tutti gli split possibili per quella variabile;
3. calcolare l'indice Gini;
4. ripetere per tutte le variabili;
5. scegliere il candidato migliore tra tutti

L'indice di gini = 0 indica che tutti gli elementi in una foglia sono della stessa categoria
Indice gini = 0.5 è il massimo di impurità che può raggiungere, indicando che ci sono 50% elementi di una categoria e 50% dell'altra

L'obiettivo è portarlo il più vicino allo 0 con lo scendere dell'albero, fino alle foglie
"""
from ucimlrepo import fetch_ucirepo
from sklearn import tree
import matplotlib.pyplot as plt

# 1) Load dataset
iris = fetch_ucirepo(id=53)

x = iris.data.features   # DataFrame con le 4 feature
y = iris.data.targets    # DataFrame con la colonna 'class'

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

plt.figure(figsize=(12, 6))
tree.plot_tree(clf, feature_names=x.columns, class_names=clf.classes_, filled=True)
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

tree.plot_tree(clf)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza: {accuracy:.2f}")

# Genera la matrice di confusione
cm = confusion_matrix(y_test, y_pred)

# 9) Visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title('Matrice di Confusione')
plt.show()



