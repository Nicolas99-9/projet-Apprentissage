#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cPickle
from pprint import pprint
import itertools
from pprint import pprint



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
        

#cree un dictionnaire normalise des patchs
dicos_normalized_patchs = generation_patches(dicos)

# dico contenant : cle data         => matrice d images, chaque ligne correspond a une image, les valeurs vont de 0 à 255
#				       chaque element est un array contenant 2 elements([ 59,  43,  50, ..., 140,  84,  72], dtype=uint8)
#                  cle labels       => liste contenant le vrai label de l image
#                  cle batch_label  => unknown
#		   cle filenames    => liste contenant le nom de l image 'camion_s_001895.png', 'trailer_truck_s_000335.png'
#
