# Hello world of machine learning ==> dataset: iris 2D with Support Vector Machine model

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import pandas as pd
import numpy as np


def print_class(n):
    if n == [0]:
        print("\tIris-setosa: class ", end="")
        print(n)
    elif n == [1]:
        print("\tIris-versicolor: class ", end="")
        print(n)
    elif n == [2]:
        print("\tIris-virginica: class ", end="")
        print(n)
    else:
        print("No class")


iris = load_iris()  # loading iris 2D dataset from sklearn
                        # carica il dataset iris 2D fornito da sklearn

print("\nThe keys that can be used inside iris[' ... '] are: ")
print(iris.keys())  # all keys
                        # tutte le chiavi contenute nel dataset

# printing dataset's raw content
    # stampa il contenuto raw del dataset

print("\n", iris['data'])  # print all flowers's values
                            # stampa tutti i valori raccolti dei vari fiori

print("\n", iris['target'])  # print classes's target
                                # stampa i target delle classi

print("\n", iris['frame'])

print("\n", iris['target_names'])  # print classes's names
                                        # stampa i nomi delle classi

print("\n", iris['DESCR'])  # dataset description
                                # stampa la descrizione del dataset

print("\n", iris['feature_names'])  # dataset's features names
                                        # nomi delle caratteristiche prese in
                                        # considerazione per i vari fiori

print("\n", iris['filename'])  # print file's name that contains the dataset
                                    # stampa il nome del file che contiene il dataset

print("\n", iris)  # print all iris
                        # stampa tutto iris

X = iris.data[:, 2:]  # assign to X flowers's features
                      # [":" = select all dataset row , "2:" = select only columns from two to the last]
                            # assegno a X le caratteristiche ==> si selezionano
                            # tutte le righe ":" dall'inizio alla fine e,
                            # delle colonne seleziono dalla seconda alla fine (le ultime due).


y = iris.target  # assign to y flower's classes: petal length, petal width
                    # assegno ad y le classi: lunghezza petalo, larghezza petalo

print("\n", iris.data)

print("\n---------------------------------------")
print("|            TRAINING-SET             |")
print("---------------------------------------\n")

print("X: features")
print(X, "\n")

print("y: classes")
print(y)

# plotting training-set data
    # plot dei punti di allenamento ==> è un rappresentazione grafica dei numeri

plt.figure(2, figsize=(8, 6))
plt.rc('grid', linestyle="-", color='black')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

x_min = y_min = 0
x_max = 7
y_max = 3
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.grid(True)
plt.show()

# algorithm fitting with data contained in dataset. Algorithm = Support Machine Vector
    # addestramento dell'algoritmo con i dati raccolti e puliti
    # uso il classificatore Support Vector Machine

clf = SVC(gamma='auto', probability=True)  # classifier
clf.fit(X, y)  # method used to learn from data

# testing-set ==> insert new data to check the validity of the model produced by algorithm
    # testing-set ==> per capire quanto è bravo questo albero, gli passo dei fiori nuovi

print("\n-----------------------------------")
print("|            TEST-SET             |")
print("-----------------------------------\n")

print("- Element [4, 1.5]: ")
print_class(clf.predict([[4, 1.5]]))  # this command is used to predict
                                      # the class of the flower with this
                                      # features: 4, 1.5
print("\tProbability:", end=" ")
print(clf.predict_proba([[4, 1.5]]))  # this command is used to
                                      # show the probability that the flower with specific features
                                      # belongs to a class

print("- Element [2, 0.5]: ")
print_class(clf.predict([[2, 0.5]]))
print("\tProbability:", end=" ")
print(clf.predict_proba([[2, 0.5]]))

print("\n", "- Element [4, 1.5]: ")
print_class(clf.predict([[4, 1.5]]))
print("\tProbability:", end=" ")
print(clf.predict_proba([[4, 1.5]]))

print("\n", "- Element [5, 1.6]: ")
print_class(clf.predict([[5, 1.6]]))
print("\tProbability:", end=" ")
print(clf.predict_proba([[5, 1.6]]))

print("\n", "- Element [2, 2]: ")
print_class(clf.predict([[2, 2]]))
print("\tProbability:", end=" ")
print(clf.predict_proba([[2, 2]]))
