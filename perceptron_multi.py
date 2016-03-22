import numpy as np
from math import sqrt
import random
import codecs

class Perceptron_multi:

    def __init__(self, liste_appren, liste_test, liste_verif, nb_class):
        self.appren = liste_appren
        self.test = liste_test
        self.verif = liste_verif
        self.nb = nb_class
        self.poid = []
        for i in range(len(self.appren)):
            self.appren[i][1] + [1]
        for i in range(len(self.test)):
            self.test[i][1] + [1]
        for i in range(len(self.verif)):
            self.verif[i][1] + [1]

    def verif_train(self):
        reussite = 0.0
        for image in self.verif:
            if self.classify(image[1]) == image[0] :
                         reussite += 1
       	return reussite/len(self.verif)

    def learn_shuffle(self, nb_phase, crit_arret, lr):
        self.poid = []
        for i in range(self.nb) :
            self.poid.append( [0 for _ in range( len(self.appren[0][1]) ) ]  )
        for i in range(nb_phase):
            erreur = 0.0
            random.shuffle(self.appren)
            for patch in self.appren:
                pred = self.classify(patch[1])
                if pred != patch[0]:
                    erreur += 1;
                    for j in range( len(self.poid[0]) ):
                        self.poid[pred][j] -= patch[1][j] * lr
                        self.poid[patch[0]][j] += patch[1][j] * lr
            lr = lr * 0.95
            print "phase : " ,i
            reussite = self.verif_train()
            print "taux erreur appren: " , erreur/len(self.appren) , "  taux reussite : " , reussite

    def learn_mean(self, nb_phase, crit_arret, lr):
        fin = 0
        sav = []
        self.poid = []
        for i in range(self.nb) :
            self.poid.append( [0 for _ in range( len(self.appren[0][1]) ) ]  )
            sav.append( [0 for _ in range( len(self.appren[0][1]) ) ]  )
        for i in range(nb_phase):
            erreur = 0.0
            random.shuffle(self.appren)
            for patch in self.appren:
                pred = self.classify(patch[1])
                if pred != patch[0]:
                    erreur += 1;
                    for j in range( len(self.poid[0]) ):
                        self.poid[pred][j] -= patch[1][j]
                        self.poid[patch[0]][j] += patch[1][j]

            inter = self.poid
            for j in range(self.nb) :
                for k in range(len(self.appren[0][1])):
                    sav[j][k] = sav[j][k] + self.poid[j][k]
                    self.poid[j][k] = self.poid[j][k] / (i+1)
            print "phase : " ,i
            print "taux erreur appren: " , (erreur/len(self.appren)) ;
            reussite = self.verif_train()
            print "taux reussite : " , reussite

            self.poid = inter
        for j in range(self.nb) :
            for k in range(len(self.appren[0][1])):
                sav[j][k] = sav[j][k] + self.poid[j][k]
                self.poid[j][k] = self.poid[j][k] / (i+1)

    def classify(self, obs):
        label,res = 0,np.dot(obs, self.poid[0])
        for i in range(1,self.nb):
            inter = np.dot(obs, self.poid[i] )
            if inter > res:
                res = inter
                label = i
        return label

    def testing(self):
        reussite = 0.0
        for image in self.test:
            if self.classify(image[1]) == image[0] :
                         reussite += 1
       	return reussite/len(self.test)
