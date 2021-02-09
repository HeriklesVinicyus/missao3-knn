from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import sklearn.svm as svm


import numpy as np


#nao-normalizado
dataset_noun = np.genfromtxt('base/abalone_nostr.csv', delimiter=',')

np.random.shuffle(dataset_noun)

Xs_n = dataset_noun[:, :-1]
Ys_n = dataset_noun[:, -1]

X_train_n, X_test_n ,y_train_n, y_test_n = train_test_split(Xs_n,Ys_n,test_size=0.33, random_state=42)

#normalizado e agrupado
dataset = np.genfromtxt('base/abalone_preprocessado.csv', delimiter=',')
treino = np.genfromtxt('base/abalone_treino.csv', delimiter=',')
teste = np.genfromtxt('base/abalone_test.csv', delimiter=',')

np.random.shuffle(dataset)

Xs = dataset[:, :-1]
Ys = dataset[:, -1]

X_train = treino[:, :-1]
y_train = treino[:, -1]
X_test = teste[:, :-1]
y_test = teste[:,-1]


clf = KNeighborsClassifier(n_neighbors=5,p=1)

clf.fit(X_train_n, y_train_n)
y_pred = clf.predict(X_test_n)
print('para os dados não normalizados')
print(classification_report(y_test_n, y_pred))

scores = cross_val_score(clf, X_train_n, y_train_n, cv=10)
print("Accuracy: {} (+/- {})".format(scores.mean(), scores.std() * 2))


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('para os dados normalizados e agrupados')
print(classification_report(y_test, y_pred))

scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Accuracy: {} (+/- {})".format(scores.mean(), scores.std() * 2))

#experimentos:
#nao normalizados
k_s = list(range(1,50, 2))
experimentos_n = []

for i in k_s:
    clf.n_neighbors = i
    scores = cross_val_score(clf, Xs_n, Ys_n, cv=10)
    experimentos_n.append((i,scores.mean(), scores.std()*2))
    #print("─ K: {}, Accuracy: {} (+/- {})".format(i,scores.mean(), scores.std() * 2))


experimentos_n.sort(key=lambda tup: tup[1])
print("\n\nExperimentos na Base Abalone não Normalizados ─ K: {}, Accuracy: {} (+/- {})".format(experimentos_n[-1][0],experimentos_n[-1][1], experimentos_n[-1][2]))

#normalizados
experimentos = []
for i in k_s:
    clf.n_neighbors = i
    scores = cross_val_score(clf, Xs, Ys, cv=10)
    experimentos.append((i,scores.mean(), scores.std()*2))
    #print("─ K: {}, Accuracy: {} (+/- {})".format(i,scores.mean(), scores.std() * 2))
experimentos.sort(key=lambda tup: tup[1])
print("Experimentos na Base Abalone Normalizados e agrupados ─ K: {}, Accuracy: {} (+/- {})".format(experimentos[-1][0],experimentos[-1][1], experimentos[-1][2]))

#Outras bases
dataset_iris = np.genfromtxt('base/iris.csv', delimiter=',')

np.random.shuffle(dataset_iris)

Xs_i = dataset_iris[:, :-1]
Ys_i = dataset_iris[:, -1]

X_train_i, X_test_i ,y_train_i, y_test_i = train_test_split(Xs_i,Ys_i,test_size=0.33, random_state=42)


clf.fit(X_train_i, y_train_i)
y_pred = clf.predict(X_test_i)


print('Base iris')
print(classification_report(y_test_i, y_pred))
scores = cross_val_score(clf, Xs_i, Ys_i, cv=3)
print("Accuracy: {} (+/- {})".format(scores.mean(), scores.std() * 2))


experimentos_iris = []

for i in k_s:
    clf.n_neighbors = i
    scores = cross_val_score(clf, Xs_i, Ys_i, cv=10)
    experimentos_iris.append((i,scores.mean(), scores.std()*2))
    #print("─ K: {}, Accuracy: {} (+/- {})".format(i,scores.mean(), scores.std() * 2))
experimentos_iris.sort(key=lambda tup: tup[1])
print("Experimentos na Base iris ─ K: {}, Accuracy: {} (+/- {})".format(experimentos_iris[-1][0],experimentos_iris[-1][1], experimentos_iris[-1][2]))


dataset_iono = np.genfromtxt('base/ionosphere_nostr.csv', delimiter=',')

np.random.shuffle(dataset_iono)

Xs__iono = dataset_iono[:, :-1]
Ys__iono = dataset_iono[:, -1]

X_train_iono, X_test_iono ,y_train__iono, y_test__iono = train_test_split(Xs__iono,Ys__iono,test_size=0.33, random_state=42)

clf.fit(X_train_iono, y_train__iono)
y_pred = clf.predict(X_test_iono)
print('Base Ionosfera')
print(classification_report(y_test__iono, y_pred))
scores = cross_val_score(clf, Xs__iono, Ys__iono, cv=3)
print("Accuracy: {} (+/- {})".format(scores.mean(), scores.std() * 2))

experimentos_iono = []

for i in k_s:
    clf.n_neighbors = i
    scores = cross_val_score(clf, Xs__iono, Ys__iono, cv=10)
    experimentos_iono.append((i,scores.mean(), scores.std()*2))
    #print("─ K: {}, Accuracy: {} (+/- {})".format(i,scores.mean(), scores.std() * 2))

experimentos_iono.sort(key=lambda tup: tup[1])
print("Experimentos na Base iono ─ K: {}, Accuracy: {} (+/- {})".format(experimentos_iono[-1][0],experimentos_iono[-1][1], experimentos_iono[-1][2]))
