#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cPickle
from pprint import pprint
import itertools
from pprint import pprint
import random
import operator
import matplotlib.pyplot as plt
import kmeans

NUMBER_CLASSES = 10

#------------------------------ K means algorithm ---------------------------------------------------

def check(dictionnary, k):
    for element in dictionnary:
        if(len(dictionnary[element])<k):
            return False
    return True


#return a dictionnary with 
def choose_initiale(data, k , labels) :
    list_initiale =  []
    taille = len(data)
    count = 0
    while(count < k ):
        tmp = list(data[count])
        list_initiale.append(tmp)
        count += 1
    return list_initiale


#return the average value of a list of vectors

def moyenne_element(liste):
    taille = len(liste)
    finale = [0 for i in range(len(liste[0]))]
    for element in liste:
        finale = np.add(finale,element)
    for i in range(len(finale)):
        finale[i] = np.divide(finale[i],taille)
    return finale
#-----------------------------------------------------------------------------------------------------

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

dicos = unpickle("cifar-10-batches-py/data_batch_1")

def show_image(img):
    to_display = []
    count = 0
    tmp = []
    for i in range(1024):
        tmp.append([img[i]/255.0,img[i+1024]/255.0,img[i+2048]/255.0])
        count +=1
        if(count%32==0):
            count = 0
            to_display.append(tmp)
            tmp = []
    to_display = np.array(to_display)
    plt.imshow(to_display)
    plt.show()


def display_image(i):
    show_image(dicos['data'][i])

#display_image(0)

dicos_test = unpickle("cifar-10-batches-py/data_batch_2")


dicos_test_value = unpickle("cifar-10-batches-py/data_batch_3")

def normalized(ma_liste):
    means = np.mean(ma_liste)
    var = np.std(ma_liste)
    for i in range(len(ma_liste)):
        ma_liste[i] = abs(((ma_liste[i])-means)/var) 
    return ma_liste



def get_patches_from_image(image):
    result = {}
    for i in range(4):
        tmp = []
        tmp = tmp  + [image[count] for count in range(256*i,256*i+256)]
        tmp = tmp  + [image[count] for count in range(256*i +1024, 256*i+256+1024)]
        tmp = tmp  + [image[count] for count in range(256*i +2048, 256*i+256+2048)]
        result[i] = tmp
    return result


#generate a dictionnary
#dictionnary with 10 differents keys, each value contains 4 dictionnaries, each one for one patch

'''
def construction_dictionnaire_n_patches(dictionnary,N):
    dico_random = choose_initiale(dictionnary['data'],N,dictionnary['labels'])
    dico_patches_moyenne = {}
    for classe in dico_random:
        dico_patches_moyenne[classe] = {}
        dico_patches_moyenne[classe][0] = []
        dico_patches_moyenne[classe][1] = []
        dico_patches_moyenne[classe][2] = []
        dico_patches_moyenne[classe][3] = []
        elements_de_la_classe = dico_random[classe]
        #for each element, slice the array into patches
        for element in elements_de_la_classe:
            patches =get_patches_from_image(element)
            #patch ok
            dico_patches_moyenne[classe][0].append(patches[0])
            dico_patches_moyenne[classe][1].append(patches[1])
            dico_patches_moyenne[classe][2].append(patches[2])
            dico_patches_moyenne[classe][3].append(patches[3])
    #dico_patches_moyenne contain N * 4 random patch selected from the data
    #compute the average
    for element in dico_patches_moyenne:
        dico_patches_moyenne[element][0] = list(moyenne_element(dico_patches_moyenne[element][0]))
        dico_patches_moyenne[element][1] = list(moyenne_element(dico_patches_moyenne[element][1]))
        dico_patches_moyenne[element][2] = list(moyenne_element(dico_patches_moyenne[element][2]))
        dico_patches_moyenne[element][3] = list(moyenne_element(dico_patches_moyenne[element][3]))
    return dico_patches_moyenne

'''


#randomly select patches, and then compute kmeans for each patches (4)
def construction_dictionnaire_n_patches(dictionnary,N):
    dico_random = choose_initiale(dictionnary['data'],N,dictionnary['labels'])
    dico_patches_moyenne = {}
    dico_patches_moyenne[0] = []  #top left
    dico_patches_moyenne[1] = []  #top right
    dico_patches_moyenne[2] = []  #bottom left
    dico_patches_moyenne[3] = []  #bottom right
    for i in range(len(dico_random)):
        patches =  get_patches_from_image(dico_random[i])
        dico_patches_moyenne[0].append(patches[0])
        dico_patches_moyenne[1].append(patches[1])
        dico_patches_moyenne[2].append(patches[2])
        dico_patches_moyenne[3].append(patches[3])
    print("DEBUT DES K MEANS")
    for i in range(4):
        print("DEBUT DES K MOYENNES ",i)
        #renvoie la partition à laquelle appartient chaque element et la liste des centres des partitions
        partitions,moyennes = kmeans.kmeans(dico_patches_moyenne[i],10,5,20)
        dico_patches_moyenne[i] =  moyennes
    print("-----------------------------------------")
    return dico_patches_moyenne




#compute the representants, each classe has 4 representants
elements_aleatoires_moyenne =  construction_dictionnaire_n_patches(dicos,len(dicos['data'])/4)

#elements_aleatoires_moyenne[0] to get the K means of the top left
#elements_aleatoires_moyenne[0][0] => get the first average value of the top left patch




#------------------- TESTS--------------------------------------------------------------------------
#calcule la distance pour chaque patch avec l'ensemble des representants puis genere la nouvelle representation des donnees



def distance(a,b):
    return np.linalg.norm(a-b)


def test_model(images,model,nb):
    labels = images['labels']
    data = images['data']
    resultat = []
    assert(nb<len(data)),("Vous ne pouvez tester que sur : " + str(len(data)))
    for element in range(nb):
        patchs_actuel = get_patches_from_image(data[element])
        buffers = []
        for i in patchs_actuel:
            #each patch
            petit = []
            var = 9999999  #distance max
            classe = -1
            for j in range(NUMBER_CLASSES):
                varss = distance(np.array(model[i][j]),np.array(patchs_actuel[i]))
                if(varss<var):
                    var = varss
                    classe = j
                    #print("j : ",j,varss,"classe :",classe)
            tableau  = [0 for i in range(NUMBER_CLASSES)]
            tableau[classe] = 1 
            print(classe," vraie lael : ",labels[element],tableau)
            buffers = buffers + tableau
        phrase = (element,buffers)
        resultat.append(phrase)
    return resultat




print("Generation de la nouvelle representation des donnes")
nouvelles_donnes = test_model(dicos_test,elements_aleatoires_moyenne,200)

#nouvelles_test = test_model(dicos_test_value,elements_aleatoires_moyenne,1000)
print("Generation terminee")





def get_positions(liste):
    tmp =  [i for i in range(len(liste)) if liste[i]==1]
    return tmp

def has_couple(x,y,ex, ey):
    for i in range(len(x)):
        if(x[i] == ex and y[i] == ey):
            return i
    return -1

def plot_model(donnees):
    labels =  dicos_test['labels'] 
    dicoss = {}
    for i in range(NUMBER_CLASSES):
        dicoss[i] = []
    for (etiquette,element) in donnees:
        dicoss[labels[etiquette]] = dicoss[labels[etiquette]]+  get_positions(element)
    x = []
    y = []
    s = []
    for i in range(NUMBER_CLASSES):
        print(len(dicoss[i]))
    for i in range(NUMBER_CLASSES):
        for element in dicoss[i]:
            val = has_couple(x,y,element,i)
            if(val==-1):
                x.append(element)
                y.append(i)
                s.append(10)
            else:
                s[val] = s[val]+15
    plt.scatter(x,y,s)
    print(len(donnees))
    plt.title('Repartition des points en fonction de la classe')
    plt.xlabel('valeurs des points')
    plt.ylabel('classe')
    plt.savefig('ScatterPlot.png')
    plt.show()

plot_model(nouvelles_donnes)


#------------------------ Perceptron utilisant les nouvelles donnes --------------------------
#

#return the estimate classes of an observation
def classify(observation, poids):
    vl = 0
    classe = -1
    for i in range(len(poids)):
        tmp = np.dot(observation,poids[i])
        if(tmp>vl):
            vl = tmp
            classe = i
    return classe

def learn(train,nb,poids,labels):
    for s in range(1,nb+1): 
        erreur = 0.0
        for (etiquette,element) in train:
            estimation = classify(element,poids) 
            reelle= labels[etiquette]
            if(not (estimation == reelle)):
                reelle= labels[etiquette]
                erreur +=1
                poids[reelle] = poids[reelle] + element
                poids[estimation] = poids[estimation] -element
        print("taux d'erreur : ",erreur/len(train))
    return poids
   
#poids = [[0 for i in range(
print("Debut du perceptron")
poids = learn(nouvelles_donnes,50,np.array([[0 for i in range(40)] for j in range(NUMBER_CLASSES)]),dicos_test['labels'])

'''
print("Valeur des poids : ")
print(poids)
print("Fin du perceptron")

def test(corpus,poids,labels):
    erreur = 0.0
    for etiquette,element in corpus:
        vm = classify(element,poids)
        print(vm , "vraie valeur : ",labels[etiquette])
        if not vm == labels[etiquette]:
            erreur +=1.0
    return erreur/len(corpus)

print("debut des tests")
print("Taux d'erreur : ",test(nouvelles_test,poids,dicos_test_value['labels']))
print("fin des test")

'''
