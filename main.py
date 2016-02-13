#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cPickle
from pprint import pprint
import itertools
from pprint import pprint
import random

NUMBER_CLASSES = 10

#------------------------------ K means algorithm ---------------------------------------------------

def check(dictionnary, k):
    for element in dictionnary:
        if(len(dictionnary[element])<k):
            return False
    return True


#return a dictionnary with 
def choose_initiale(data, k , labels) :
    list_initiale =  {}
    for i in range(NUMBER_CLASSES):
        list_initiale[i] = []
    taille = len(data)
    while(not check(list_initiale,k)):
        i  = random.randrange(taille)
        tmp = list(data[i])
        label = labels[i]
        if (not (tmp in list_initiale[label]) and (len(list_initiale[label])< k )) :
            list_initiale[label].append(tmp)
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
        tmp = tmp  + list((itertools.islice(image, 256*i, 256*i+256)))
        tmp = tmp  + list((itertools.islice(image, 256*i +1024, 256*i+256+1024)))
        tmp = tmp  + list((itertools.islice(image, 256*i +2048, 256*i+256+2048)))
        result[i] = tmp
    return result


def generation_patches(dicos):
    finale = {}
    count = 0
    for image in dicos['data'][:1]:
        #len : 3072 : (1024 R , 1024 G , 1024 B)
        #256 elements par patch 
        finale[count] = []
        for i in range(4):
            tmp = []
            tmp = tmp  + list((itertools.islice(image, 256*i, 256*i+256)))
            tmp = tmp  + list((itertools.islice(image, 256*i +1024, 256*i+256+1024)))
            tmp = tmp  + list((itertools.islice(image, 256*i +2048, 256*i+256+2048)))
            finale[count].append(tmp)
        count +=1
    for element in finale:
        finale[element] = [normalized(finale[element][i]) for i in range(len(finale[element]))]
    return finale
        

#generate a dictionnary
#dictionnary with 10 differents keys, each value contains 4 dictionnaries, each one for one patch
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
        for element in elements_de_la_classe:
            patches =get_patches_from_image(element)
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


elements_aleatoires_moyenne =  construction_dictionnaire_n_patches(dicos,2)

#cree un dictionnaire normalise des patchs
#dicos_normalized_patchs = generation_patches(dicos)


#construction_dictionnaire_n_patches(dicos)


# dico contenant : cle data         => matrice d images, chaque ligne correspond a une image, les valeurs vont de 0 à 255
#				       chaque element est un array contenant 2 elements([ 59,  43,  50, ..., 140,  84,  72], dtype=uint8)
#                  cle labels       => liste contenant le vrai label de l image
#                  cle batch_label  => unknown
#		   cle filenames    => liste contenant le nom de l image 'camion_s_001895.png', 'trailer_truck_s_000335.png'
#
