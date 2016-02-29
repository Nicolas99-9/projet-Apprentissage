#coding=utf-8
from __future__ import division
from __future__ import print_function
from random import choice
import numpy as np
from pprint import pprint
import operator
import matplotlib.pyplot as plt
        
def choose_initiale(data, k) :
    list_initiale = []
    while(len(list_initiale)!=k):
        tmp = choice(data)
        if not tmp in list_initiale :
            list_initiale.append(tmp)
    return list_initiale


def sortes(ma_list):
    result = sorted(ma_list.items(), key=operator.itemgetter(1))
    return result


def distance (x,y) :
	return np.linalg.norm(np.array(x) - np.array(y))

def moyenne_element(liste):
    return np.mean(liste,axis=0)

def kmeans(data, k, t, maxiter):
    points = choose_initiale(data, k)
    count = 0
    plop = len(data[0])
    dict_result = {}
    finale = {}
    sigma = 0.0
    finale = {}
    for point in data :
        dict_result[tuple(point)] = -1
    for i in range(len(points)):
        dict_result[tuple(points[i])] = i
        finale[i] = points[i] 
    error = 9999
    nbIter = 1
    while(error>t and nbIter < maxiter):
        nbIter+= 1
        print("iteration : ",nbIter)
        classes = {}
        for i in range(k):
            classes[i] = []
        for element in data:
            tmp = {}
            for classe in range(k):
                tmp[classe] = distance(finale[classe],element)
            tmp = sortes(tmp)
            cle = tmp[0][0]
            valeur = tmp[0][1]
            dict_result[tuple(element)] = cle
            classes[cle].append(element)
        for classe in range(k):
            finale[classe] = moyenne_element(classes[classe])
        taux_Erreur = 0.0
        for element in data:
            taux_Erreur += distance(finale[dict_result[tuple(element)]],element)
        error  = abs(taux_Erreur-error)
    tab1 = []
    for element in dict_result:
        tab1.append((list(element),dict_result[element]))
    tab2 = []
    for element in finale:
        tab2.append((list(finale[element])))
    return (tab1,tab2)
		

#res = kmeans(data, 2, 1, 12)

'''
tabs = [[1,2],[0,4],[2,5],[5,5],[6,4],[0,2],[1,1]]
print(kmeans(tabs,3,1,50))


X = [i[0] for i in tabs]
Y = [i[1] for i in tabs]

plt.scatter(X,Y)
plt.title('Repartition des points en fonction de la classe')
plt.xlabel('valeurs des points')
plt.ylabel('classe')
plt.show()
'''
