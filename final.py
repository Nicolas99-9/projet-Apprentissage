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
from sklearn.feature_extraction import image




NUMBER_CLASSES = 100
#number of patches per image
NUMBER_PATCHES =  4
#number of perdios for k means function
NUMBER_PERIODS = 5
#number of images to use for  the k means function
NUMBER_K_MEANS = 1
#number of data for training
NUMBER_TRAIN = 5000
#number of data to evaluate our model
NUMBER_TEST = 5000
#number of periods of the perceptron
EPOQS = 100


#------------------------------ K means algorithm ---------------------------------------------------


#return a dictionnary with k first elements of the liste : list(list())
def choose_initiale(data, k , labels) :
    list_initiale =  []
    taille = len(data)
    count = 0
    while(count < k ):
        tmp = list(data[count])
        list_initiale.append(tmp)
        count += 1
    return list_initiale


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

def show_image_after(img):
    to_display = []
    taile = np.sqrt(NUMBER_PATCHES)
    taile = int(32/taile)*3
    print("tailel : ",taile)
    for i in xrange(0,len(img),taile):
        tmp = []
        print(i)
        for j in xrange(i,i+taile,3):
            tmp.append([img[j]/255.0,img[j+1]/255.0,img[j+2]/255.0])
        to_display.append(tmp)
    to_display = np.array(to_display)
    plt.imshow(to_display)
    plt.show()

#function to display the real image
def display_image(i,dicos):
    show_image(dicos['data'][i])

#load the file of the data
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#function to normalized a list
def normalized(ma_liste):
    means = np.mean(ma_liste)
    var = np.std(ma_liste)
    for i in range(len(ma_liste)):
        ma_liste[i] = abs(((ma_liste[i])-means)/var) 
    return ma_liste


#cut an image into patches
'''
def get_patches_from_image(image):
    result = {}
    param = 1024/NUMBER_PATCHES
    for i in range(NUMBER_PATCHES):
        tmp = []
        tmp = tmp  + [image[count] for count in range(param*i,param*i+param)]
        tmp = tmp  + [image[count] for count in range(param*i +1024, param*i+param+1024)]
        tmp = tmp  + [image[count] for count in range(param*i +2048, param*i+param+2048)]
        result[i] = tmp
    return result
'''
def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

'''
def get_patches_from_image(image2):
    result = {}
    for i in range(NUMBER_PATCHES):
        result[i] = []
    tmp = []
    for i in range(1024):
        tmp.append(image2[i])
        tmp.append(image2[i+1024])
        tmp.append(image2[i+2048])
    tmp = np.array(tmp)
    tmp = tmp.reshape(32,96)
    taille = 96/np.sqrt(NUMBER_PATCHES)
    taille2 = 32/np.sqrt(NUMBER_PATCHES)
    print(taille,taille2)
    patches =  image.extract_patches_2d(tmp, (int(taille2), int(taille)))
    print(len(patches))
    return result
'''




'''
def get_patches_from_image(image):
    result = {}
    for i in range(NUMBER_PATCHES):
        result[i] = []
    #nb de colonnes
    nb = int(np.sqrt(NUMBER_PATCHES))
    taille_ligne = 32
    par_ligne = int(taille_ligne/nb)
    splied = list(chunks(image,par_ligne))
    count = 0
    tmp = 0
    k = 0
    tier = len(splied)/3
    doubles = 2*tier
    for i in range(len(splied)/3):
        indice = (i%nb)+k
        if(indice>=NUMBER_PATCHES):
            return result
        ddd = []
        ddd.append(iter(splied[i]))
        ddd.append(iter(splied[i+tier]))
        ddd.append(iter(splied[i+doubles]))
        #print("longuiedueuueueu",len(splied[i+doubles]),indice)
        result[indice] = result[indice]+ list(it.next() for it in itertools.cycle(ddd))
        if((indice-k)==(nb-1)):
            tmp+=1
        if(tmp==(par_ligne)):
            k += nb
            tmp =  0
    return result
''''''
import numpy as numpy
def get_patches_from_image(image):
    size_r = 5
    size = 5
    for r in range(0,test_image.shape[0] - windowsize_r, windowsize_r):
        for c in range(0,test_image.shape[1] - windowsize_c, windowsize_c):
            window = test_image[r:r+windowsize_r,c:c+windowsize_c]
            hist = numpy.histogram(window,bins=grey_levels)
'''


#randomly select patches, and then compute kmeans for each patches (4)
def construction_dictionnaire_n_patches(dictionnary,N):
    dico_random = choose_initiale(dictionnary['data'],N,dictionnary['labels'])
    dico_patches_moyenne = {}
    for i in range(NUMBER_PATCHES):
        dico_patches_moyenne[i] = []
    for i in range(len(dico_random)):
        patches =  get_patches_from_image(dico_random[i])
        for j in range(NUMBER_PATCHES):
            dico_patches_moyenne[j].append(patches[j])
    print("DEBUT DES K MEANS")
    for i in range(NUMBER_PATCHES):
        #renvoie la partition à laquelle appartient chaque element et la liste des centres des partitions
        partitions,moyennes = kmeans.kmeans(dico_patches_moyenne[i],NUMBER_CLASSES,1,NUMBER_PERIODS)
        dico_patches_moyenne[i] =  moyennes
    print("-----------------------------------------")
    return dico_patches_moyenne

#return the eucldian distance between two vectors
def distance(a,b):
    return np.linalg.norm(a-b)



dicos = unpickle("cifar-10-batches-py/data_batch_1")
elements_aleatoires_moyenne =  construction_dictionnaire_n_patches(dicos,NUMBER_K_MEANS)



#display_image(0,dicos)

show_image_after(get_patches_from_image(dicos['data'][0])[0])
show_image_after(get_patches_from_image(dicos['data'][0])[1])
show_image_after(get_patches_from_image(dicos['data'][0])[2])
show_image_after(get_patches_from_image(dicos['data'][0])[3])




dicos_test = unpickle("cifar-10-batches-py/data_batch_2")
dicos_test_value = unpickle("cifar-10-batches-py/data_batch_3")
#---------------------------------------FEATURES GENERATION--------------------------------


def test_model(images,model,nb):
    labels = images['labels']
    data = images['data']
    resultat = []
    for element in range(nb):
        patchs_actuel = get_patches_from_image(data[element])
        buffers = []
        for i in patchs_actuel:
            #each patch
            petit = []
            var = 9999999  #distance max
            classe = -1
            for j in range(NUMBER_CLASSES):
                for w in range(4):
                    varss = distance(np.array(model[w][j]),np.array(patchs_actuel[i]))
                    if(varss<var):
                        var = varss
                        classe = j
                    #print("j : ",j,varss,"classe :",classe)
            tableau  = [0 for i in range(NUMBER_CLASSES)]
            tableau[classe] = 1 
            #print(classe," vraie lael : ",labels[element],tableau)
            buffers = buffers + tableau
        phrase = (element,buffers)
        resultat.append(phrase)
    return resultat




print("Generation de la nouvelle representation des donnes")
nouvelles_donnes = test_model(dicos_test,elements_aleatoires_moyenne,NUMBER_TRAIN)
nouvelles_test = test_model(dicos_test_value,elements_aleatoires_moyenne,NUMBER_TEST)
print("Generation terminee")




#-------------------------------- FUNCTIONS TO PLOT THE ELEMENTS------------------------
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


def test(corpus,poids,labels):
    erreur = 0.0
    for etiquette,element in corpus:
        vm = classify(element,poids)
        #print(vm , "vraie valeur : ",labels[etiquette])
        if not vm == labels[etiquette]:
            erreur +=1.0
    return erreur/len(corpus)



print("Debut du perceptron")
poids = learn(nouvelles_donnes,EPOQS,np.array([[0 for i in range(NUMBER_PATCHES*NUMBER_CLASSES)] for j in range(NUMBER_CLASSES)]),dicos_test['labels'])
print("Valeur des poids : ")
print(poids)
print("Fin du perceptron")
print("debut des tests")
print("Taux d'erreur : ",test(nouvelles_test,poids,dicos_test_value['labels']))
print("fin des test")


