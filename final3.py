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
import kmeans2
from scipy.cluster.vq import whiten
from patcher import Patcher
from sklearn import cluster
import pickle
from sklearn import metrics
from sklearn import cross_validation
from sklearn import svm
from perceptron_multi import Perceptron_multi



'''
NUMBER_CLASSES = 700
#number of patches per image
NUMBER_PATCHES =  4
#number of perdios for k means function
NUMBER_PERIODS = 11
#number of images to use for  the k means function
NUMBER_K_MEANS = 28000
#number of data for training
NUMBER_TRAIN = 28000
#number of data to evaluate our model
NUMBER_TEST = 1500
#number of periods of the perceptron
EPOQS = 60
'''






#Parameters to test the algorithm

'''
NUMBER_CLASSES = 250
#number of patches per image
NUMBER_PATCHES =  4
#number of perdios for k means function
NUMBER_PERIODS = 10
#number of images to use for  the k means function
NUMBER_K_MEANS = 1800
#number of data for training
NUMBER_TRAIN = 8000
#number of data to evaluate our model
NUMBER_TEST = 1800
#number of periods of the perceptron
EPOQS = 60
'''


'''
NUMBER_CLASSES = 150
#number of patches per image
NUMBER_PATCHES =  4
#number of perdios for k means function
NUMBER_PERIODS = 10
#number of images to use for  the k means function
NUMBER_K_MEANS = 5000
#number of data for training
NUMBER_TRAIN = 5000
#number of data to evaluate our model
NUMBER_TEST = 800
#number of periods of the perceptron
EPOQS = 60
'''

'''
NUMBER_CLASSES = 1600
#size of each patch
NUMBER_PATCHES =  16
#number of perdios for k means function
NUMBER_PERIODS = 10
#number of images to use for  the k means function
NUMBER_K_MEANS = 11000
#number of data for training
NUMBER_TRAIN = 8000
#number of data to evaluate our model
NUMBER_TEST = 1300
#number of periods of the perceptron
EPOQS = 40
'''



NUMBER_CLASSES = 50
#size of each patch
NUMBER_PATCHES =  16
#number of perdios for k means function
NUMBER_PERIODS = 10
#number of images to use for  the k means function
NUMBER_K_MEANS = 300
#number of data for training
NUMBER_TRAIN = 300
#number of data to evaluate our model
NUMBER_TEST = 200
#number of periods of the perceptron
EPOQS = 40


#profiling :python -m cProfile final2.py

'''
1) presentation sujet & traitement langage
2) methode utilise (details techniques) , libraires et codes
3) deroulement et methode de travail
4) resultats et analyse


#convulationnal => 
#stride
#kmeans initilisation
redecouper en 4 les patches

'''
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

#show an image
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


#function to display the real image
def display_image(i,dicos):
    show_image(dicos['data'][i])

#load the file of the data
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


#normalized a patch or an image
def normalized(ma_liste):
    means = np.mean(ma_liste)
    var = np.var(ma_liste)
    for i in range(len(ma_liste)):
        ma_liste[i] = (((ma_liste[i])-means)/var)
    return ma_liste

#sauvegarde un dictionnaire
def save_element(filename,dico):
    with open(filename, 'wb') as handle:
        pickle.dump(dico, handle)

#charge un dictionnaire ou une liste
def load_element(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def debug_image(partitions):
    dics = {}
    for element in partitions:
        points,val = element
        if(val in dics):
            dics[val].append(points)
        else:
            dics[val] = []
            dics[val].append(points)
    print("dico 0")
    for i in dics[0]:
            patcher.show_patches(np.array(i))
    print("dico  2")
    for i in dics[2]:
            patcher.show_patches(np.array(i))
    print("dico  9")
    for i in dics[9]:
            patcher.show_patches(np.array(i))
    

#construit le dictionnaire des centroids
def construction_dictionnaire_n_patches(dictionnary,N):
    dico_random = choose_initiale(dictionnary['data'],N,dictionnary['labels'])
    mes_patches = []
    for i in range(len(dico_random)):
        patches =  patcher.get_patches_from_image(dico_random[i])
        for j in range(len(patches)):
            tmp = normalized(np.array(patches[j]).astype(float))
            if(np.isnan(tmp).any()):
                #print("NAN ",i,j,tmp)
                mes_patches.append(np.zeros(len(tmp)))
            else:
                mes_patches.append(tmp)
    #random.seed(45)
    random.shuffle(mes_patches)
    #partitions,moyennes = kmeans2.kmeans(mes_patches,NUMBER_CLASSES,150,NUMBER_PERIODS)
    #n_init = 3
    kmeans = cluster.KMeans(n_clusters=NUMBER_CLASSES,n_init = 1 , verbose= True, max_iter = 100, n_jobs=-1)
    kmeans.fit(mes_patches)
    centroids = kmeans.cluster_centers_
    #debug_image(partitions)
    return centroids


#fusionne les dictionnaires de données contenant les images
def merge_dicos(dicos1 , dicos2 , dicos3):
    print(dicos1['data'],dicos2['data'])
    for k in dicos2.keys():
        if(k != "data"):
            print("before " , len(dicos1[k]))
            dicos1[k] = dicos1[k] + dicos2[k] + dicos3[k]
            print( k ,"after " , len(dicos1[k]))
        else:
            dicos1[k] = np.array(list(dicos1[k]) + list(dicos2[k]) + list(dicos3[k]))
    return dicos1

dicos = merge_dicos( unpickle("cifar-10-batches-py/data_batch_1"),unpickle("cifar-10-batches-py/data_batch_2"),unpickle("cifar-10-batches-py/data_batch_3"))
print(dicos)


#generation des clusters


#cree un patcher pour extraire les patchs d'une image
#NUMBER_PATCHES : size of each patch, 32 : size of the image, 4:stride size
patcher = Patcher(NUMBER_PATCHES,32,4)
print("longueur de dicos  2222:",len(dicos['data']))
elements_aleatoires_moyenne =  construction_dictionnaire_n_patches(dicos,NUMBER_K_MEANS)
#elements_aleatoires_moyenne = load_element("cifar-10-batches-py-dicofull-8-test-1600")
print("taille des moyennes :",elements_aleatoires_moyenne)
#save the clusters extracted with the kmeans algorithm
#save_element("cifar-10-batches-py-dicofull-8-test-1600",elements_aleatoires_moyenne)


#---------------------------------------FEATURES GENERATION--------------------------------



#return the eucldian distance between two vectors
def distance(a,b):
    return np.linalg.norm(a-b)


#cree une liste de la nouvelle représentation des données, de la forme [0,0,1...]
def test_model(images,model,nb):
    labels = images['labels']
    data = images['data']
    resultat = []
    for element in range(nb):
        if(element%100==0):
            print(element)
        #print("element : ",element)
        patchs_actuel = patcher.get_patches_from_image(data[element])
        for i in range(len(patchs_actuel)):
            patchs_actuel[i]  = normalized(np.array(patchs_actuel[i]).astype(float))
        buffers = []
        for i in range(len(patchs_actuel)):
            #each patch
            petit = []
            var = 9999999  #distance max
            classe = -1
            for j in range(NUMBER_CLASSES):
                varss = distance(np.array(model[j]),np.array(patchs_actuel[i]))
                if(varss<var):
                    var = varss
                    classe = j
                    #print("j : ",j,varss,"classe :",classe)
            tableau  = [0 for p in range(NUMBER_CLASSES)]
            tableau[classe] = 1 
            buffers = buffers + tableau
        phrase = (element,buffers)
        resultat.append(phrase)  
    return resultat


print("Generation de la nouvelle representation des donnes")
nouvelles_donnes = test_model(dicos,elements_aleatoires_moyenne,NUMBER_TRAIN)
#save_element("save/nouvellesdonnes"+str(NUMBER_CLASSES)+"-"+str(NUMBER_CLASSES)+"-"+str(NUMBER_PATCHES)+"-"+str(NUMBER_TRAIN),nouvelles_donnes)
#nouvelles_donnes = load_element("save/nouvellesdonnes"+str(NUMBER_CLASSES)+"-"+str(NUMBER_CLASSES)+"-"+str(NUMBER_PATCHES)+"-"+str(NUMBER_TRAIN))
print("Generation terminee & sauvegarde faite")




#-------------------------------- FUNCTIONS TO PLOT THE ELEMENTS------------------------
def get_positions(liste):
    tmp =  [i for i in range(len(liste)) if liste[i]==1]
    return tmp

def has_couple(x,y,ex, ey):
    for i in range(len(x)):
        if(x[i] == ex and y[i] == ey):
            return i
    return -1

def plot_model(donnees,filename,dicos_test):
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
                s[val] = s[val]+10
    plt.scatter(x,y,s)
    print(len(donnees))
    plt.title('Repartition des points en fonction de la classe')
    plt.xlabel('valeurs des points')
    plt.ylabel('classe')
    plt.savefig(filename)
    plt.show()

#plot_model(nouvelles_donnes,'ScatterPlot.png',dicos)


#load the test data
pour_tests = unpickle("cifar-10-batches-py/data_batch_4")
test_data = test_model(pour_tests ,elements_aleatoires_moyenne,NUMBER_TEST)
#save_element("save/test_data"+str(NUMBER_CLASSES)+"-"+str(NUMBER_CLASSES)+"-"+str(NUMBER_PATCHES)+"-"+str(NUMBER_TEST),test_data)
#test_data = load_element("save/test_data"+str(NUMBER_CLASSES)+"-"+str(NUMBER_CLASSES)+"-"+str(NUMBER_PATCHES)+"-"+str(NUMBER_TEST))

print("Generation terminee & sauvegarde faite")
#--------------------------------K nearest neighbours----------------------------------------------


from sklearn import neighbors

#use the k nearest neighbours pour trouver la classe des données
def learn_k_nearest(data_train,data_test):
    donnes = [element for etiquette,element in data_train]
    eti = [dicos['labels'][etiquette] for etiquette,element in data_train]
    labels  = pour_tests['labels']
    #partitions,moyennes = kmeans2.kmeans(donnes,10,1,50)
    #estimation = [-1 for i in range(10)]
    #counter = {}
    knn = neighbors.KNeighborsClassifier()
    print("debut apprentissage")
    knn.fit(np.array(donnes), np.array(eti))
    print("fin apprentissage")
    to_predict = [element for etiquette,element in data_test]
    real_label = [pour_tests['labels'][etiquette] for etiquette,element in data_test]
    print("debut predictions")
    predictions = knn.predict(np.array(to_predict))
    print("fin predictions")
    taux_erreur = 0.0
    for i in range(len(predictions)):
        if(predictions[i]!=real_label[i]):
            taux_erreur +=1.0
    print("taux d'erreurs pour Knearest ", taux_erreur/float(len(predictions)))



#moyennes = learn_k_nearest(nouvelles_donnes,test_data)

#------------------------- SVM LEARNING with cross validation------------------------------------------------------- 
#simple svm without cross validation
def learn_SVM(data_train,data_test):
    donnes = [element for etiquette,element in data_train]
    eti = [dicos['labels'][etiquette] for etiquette,element in data_train]
    labels  = pour_tests['labels']
    svc = svm.SVC(kernel='linear',C=400, verbose=False)
    scores = cross_validation.cross_val_score(svc,donnes,eti)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#moyennes = learn_SVM(nouvelles_donnes,test_data)




#------------------------- SVM LEARNING-------------------------------------------------------

from sklearn import svm


#simple svm without cross validation
def learn_SVM(data_train,data_test,maxs):
    donnes = [element for etiquette,element in data_train]
    eti = [dicos['labels'][etiquette] for etiquette,element in data_train]
    labels  = pour_tests['labels']
    #linear
    svc = svm.SVC(kernel='linear',C=400, verbose=False,max_iter=maxs)
    print("debut apprentissage linear score : ")
    svc.fit(np.array(donnes), np.array(eti))
    print("score d'apprentissage : ",svc.score(np.array(donnes), np.array(eti)))
    print("fin apprentissage linear")
    to_predict = [element for etiquette,element in data_test]
    real_label = [pour_tests['labels'][etiquette] for etiquette,element in data_test]
    print("debut predictions")
    predictions = svc.predict(np.array(to_predict))
    print("fin predictions")
    taux_erreur = 0.0
    for i in range(len(predictions)):
        #print(predictions[i],real_label[i],pour_tests['filenames'][i])
        if(predictions[i]!=real_label[i]):
            taux_erreur +=1.0
    print("taux d'erreurs en lineaire ",maxs,taux_erreur,len(predictions) , taux_erreur/len(predictions),  taux_erreur/float(len(predictions)))




'''
moyennes = learn_SVM(nouvelles_donnes,test_data,30)
moyennes = learn_SVM(nouvelles_donnes,test_data,60)
moyennes = learn_SVM(nouvelles_donnes,test_data,199)
moyennes = learn_SVM(nouvelles_donnes,test_data,400)
moyennes = learn_SVM(nouvelles_donnes,test_data,700)
moyennes = learn_SVM(nouvelles_donnes,test_data,1000)
moyennes = learn_SVM(nouvelles_donnes,test_data,1500)
'''

#------------------------ Perceptron multiclasse utilisant les nouvelles donnes --------------------------


nb_l = int(NUMBER_TRAIN*0.80)
new_trains = zip(dicos['labels'][:nb_l],[b for a,b in nouvelles_donnes][:nb_l])
cross_valid = zip(dicos['labels'][nb_l:],[b for a,b in nouvelles_donnes][nb_l:])
new_tests = zip(pour_tests['labels'],[b for a,b in test_data])


print(new_tests)

perceptron = Perceptron_multi(new_trains,new_tests,cross_valid,10)

#nb epoqs, crit arret , learning rate
perceptron.learn_shuffle(20,0.1,0.7)
print("Taux d'erruers en tests : ", perceptron.testing())
'''
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
        #display_image(etiquette,dicos)
        if not vm == labels[etiquette]:
            erreur +=1.0
    return erreur/len(corpus)



print("Debut du perceptron")
poids = learn(nouvelles_donnes,EPOQS,np.array([[0 for i in range(NUMBER_CLASSES*16)] for j in range(10)]),dicos['labels'])
print("Taux d'erreur : ",test(test_data,poids,pour_tests['labels']))
print("fin des test")
'''
