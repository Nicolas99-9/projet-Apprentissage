#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cPickle
from pprint import pprint


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


dicos = unpickle("cifar-10-batches-py/data_batch_1")


def generation_patches(dicos):
    finale = {}
    for image in dicos['data']:
        #len : 3072 : (1024 R , 1024 G , 1024 B)
        #256 elements par patch 
        

generation_patches(dicos)

# dico contenant : cle data         => matrice d images, chaque ligne correspond a une image, les valeurs vont de 0 à 255
#				       chaque element est un array contenant 2 elements([ 59,  43,  50, ..., 140,  84,  72], dtype=uint8)
#                  cle labels       => liste contenant le vrai label de l image
#                  cle batch_label  => unknown
#		   cle filenames    => liste contenant le nom de l image 'camion_s_001895.png', 'trailer_truck_s_000335.png'
#
