# classifier evaluation ==> SVM vs RF
# the goodness of the algorithm can be verified through

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

iris = load_iris()  # loading iris 2D dataset from sklearn
                        # di tutti i dataset compresi in sklearn, carica il dataset iris

X = iris.data[:, 2:]  # assign to X flowers's features
                      # [":" = select all dataset row , "2:" = select only columns from two to the last]
                            # assegno a X le caratteristiche ==> si selezionano
                            # tutte le righe ":" dall'inizio alla fine e,
                            # delle colonne seleziono dalla seconda alla fine (le ultime due).


y = iris.target  # assign to y flower's classes: petal length, petal width
                    # assegno ad y le classi: lunghezza petalo, larghezza petalo

# split data between training-set e test-set in balanced way:
    # divide i dati bene e, in maniera bilanciata

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# random_state = 42 -> see "The Hitchhiker's Guide to the Galaxy"

print("X_train len: " + str(len(X_train)), "y_train len: " + str(len(y_train)))
print("X_test len: " + str(len(X_test)), "y_test len: " + str(len(y_test)))

clf = tree.DecisionTreeClassifier()  # Random forest classifier
                                        # classificatore Random Forest

clf.fit(X_train, y_train)  # classifier fitting only with training-data
                                # addestramento solo sul training set


clf1 = SVC(gamma='auto', probability=True)  # Support Vector Machine classifier
                                                # classificatore SVM

clf1.fit(X_train, y_train)  # classifier fitting only with training-data
                                # addestramento solo sul training set

# Classifier Evaluation:
    # VALUTAZIONE DEI CLASSIFICATORI

# clf.score() or clf1.score() is a method used to check how many mistakes the classifier makes
    # con clf.score() o clf1.score() si controlla quanti errori commette il classificatore

print("\nRF: ")
print("TRAIN-SET: ", clf.score(X_train, y_train), "error: ", (1 - clf.score(X_train, y_train)) * 100, "%")
print("TEST-SET: ", clf.score(X_test, y_test), "error: ", (1 - clf.score(X_test, y_test)) * 100, "%")

print("\nSVM: ")
print("TRAIN-SET: ", clf1.score(X_train, y_train), "error: ", (1 - clf1.score(X_train, y_train)) * 100, "%")
print("TEST-SET: ", clf1.score(X_test, y_test), "error: ", (1 - clf1.score(X_test, y_test)) * 100, "%")

if (1 - clf.score(X_test, y_test)) * 100 < (1 - clf1.score(X_test, y_test)) * 100:
    print("\nRF works better.")
elif (1 - clf.score(X_test, y_test)) == (1 - clf1.score(X_test, y_test)) * 100:
    print("\nRF and SMV are equivalent.")
else:
    print("\nSVM works better.")

# Mostra dove sono stati commessi gli errori

print("\nErrors in training-set RF: ")

predictions = clf.predict(X_train)

for elem, prediction, label in zip(X_train, predictions, y_train):
    if prediction != label:
        print(elem, " has been classified as ", prediction, " and should be ", label)

print("\nErrors in test-set RF: ")

predictions = clf.predict(X_test)

for elem, prediction, label in zip(X_test, predictions, y_test):
    if prediction != label:
        print(elem, " has been classified as ", prediction, " and should be ", label)

print("\nErrors in training-set SVM: ")

predictions = clf1.predict(X_train)

for elem, prediction, label in zip(X_train, predictions, y_train):
    if prediction != label:
        print(elem, " has been classified as ", prediction, " and should be ", label)

print("\nErrors in test-set SVM: ")

predictions = clf1.predict(X_test)

for elem, prediction, label in zip(X_test, predictions, y_test):
    if prediction != label:
        print(elem, " has been classified as ", prediction, " and should be ", label)

# CONFUSION MATRIX also known as ERROR MATRIX:
    # it is a specific table layout that allows visualization of the performance of an algorithm.
    # Each row of the matrix represents the instances in a predicted class while each column represents
    # the instances in an actual class
        # matrice di confusione --> mostra quanti errori in ogni classe.
        # se una classe è molto confusa cioè, ci sono pochi zeri nella riga,
        # allora vuol dire che i dati sono ancora grezzi ed occorre ripulirli poichè
        # l'algoritmo fa fatica ad apprendere

cm = confusion_matrix(y_train, clf.predict(X_train))

print("\nCM per Training set RF: \n", cm)

cm = confusion_matrix(y_test, clf.predict(X_test))

print("\nCM per Test set RF: \n", cm)

cm1 = confusion_matrix(y_train, clf1.predict(X_train))

print("\nCM per Training set SVM: \n", cm1)

cm1 = confusion_matrix(y_test, clf1.predict(X_test))

print("\nCM per Test set SVM: \n", cm1)

# cross validation --> quando si hanno pochi dati a turno i dati fanno parte del
# training-set e test-set

print("\nRF: ", cross_val_score(clf, X, y, cv=4))

sum_ = 0

for elem in cross_val_score(clf, X, y, cv=4):
    sum_ += elem

print("\nAccuracy: ", sum_ / 4)

print("\nSVM: ", cross_val_score(clf1, X, y, cv=4))

sum_ = 0

for elem in cross_val_score(clf1, X, y, cv=4):
    sum_ += elem

print("\nAccuracy: ", sum_ / 4)