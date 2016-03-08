import numpy as np
import cPickle
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.cluster.vq import whiten

class Patcher:
 
    def __init__(self,size_patch,imageSize):
        self.imageSize = imageSize
        self.sizeSquare = size_patch
        self.number_patches = (imageSize/size_patch)
        self.generate_elements()

    def show_informations(self):
        print("Informations about the patcher : ")
        print("Image size : ",self.imageSize)
        print("Size square :",self.sizeSquare)
        print("Number of patches per line : ", self.number_patches)
        print("Number of patches in the image : ",  self.number_patches*self.number_patches) 


    def generate_elements(self):
        self.conversion = {}
        self.conversion2 = {}
        count = 0
        for i in range(256):
            for j in range(256):
                for w in range(256):
                    self.conversion[(i,j,w)] = count
                    self.conversion2[count] = (i,j,w)
                    count+=1

    def get_patches_from_image(self,image):
        first = image[:1024]
        second = image[1024:2048]
        last = image[2048:]
        image = []
        for i in range(1024):
            image.append(first[i])
            image.append(second[i])
            image.append(last[i])
        after = np.array(image).reshape((self.imageSize,self.imageSize*3))
        patches= []
        for i in range(self.number_patches):
            for j in xrange(0,self.number_patches):
                tmp = after[i*self.sizeSquare:i*self.sizeSquare+self.sizeSquare,j*self.sizeSquare*3:j*self.sizeSquare*3+self.sizeSquare*3]
                patches.append(tmp.reshape(self.sizeSquare*self.sizeSquare*3))
        return list(np.array(patches).astype('int'))


    def get_patches_from_image_reduced(self,image):
        first = image[:1024]
        second = image[1024:2048]
        last = image[1024:]
        image = []
        for i in range(1024):
            image.append(first[i])
            image.append(second[i])
            image.append(last[i])
        after = np.array(image).reshape((self.imageSize,self.imageSize*3))
        patches= []
        for i in range(self.number_patches):
            for j in xrange(0,self.number_patches):
                tmp = after[i*self.sizeSquare:i*self.sizeSquare+self.sizeSquare,j*self.sizeSquare*3:j*self.sizeSquare*3+self.sizeSquare*3]
                patches.append(tmp.reshape(self.sizeSquare*self.sizeSquare*3))
        new_patches = list(patches)
        tmp = []
        for i in range(len(new_patches)):
            to_add = []
            for j in xrange(0,len(new_patches[i]),3):
                to_add.append(self.conversion[new_patches[i][j],new_patches[i][j+1],new_patches[i][j+2]])
            tmp.append(to_add) 
        return tmp

    def show_patches(self,patches):
        #print("l1",len(patches),len(patches[0]))
        tmp = np.array(patches)
        to_display = []
        to_add = []
        count =0
        for j in xrange(0,len(patches),3):
            to_add.append([tmp[j]/255.0  ,tmp[j+1]/255.0  ,tmp[j+2]/255.0 ])
            count +=1
            if(count%(self.sizeSquare)==0):
                to_display.append(to_add)
                to_add = [] 
                count =0
        to_display = np.array(to_display)
        plt.imshow(to_display)
        plt.show()

    def show_patches_reduced(self,patches):
        tmp = np.array(patches)
        to_display = []
        to_add = []
        count =0
        for j in xrange(0,len(patches)):
            to_add.append(list(self.conversion2[patches[j]]))
            count +=1
            if(count%(self.sizeSquare)==0):
                to_display.append(to_add)
                to_add = [] 
                count =0
        to_display = np.array(to_display)
        plt.imshow(to_display)
        plt.show()
    
    def debug(self,images,patches):
        for i in range(20):
            print("images : ",images[i],images[i+1024],images[i+2048] , " patch : ", patches[0][i],patches[0][i+1],patches[0][i+2])

    def show_image(self,img):
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


def normalized(ma_liste):
    means = np.mean(ma_liste)
    var = np.std(ma_liste)
    for i in range(len(ma_liste)):
        ma_liste[i] = (((ma_liste[i])-means)/var) 
    return ma_liste

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


'''
dicos = unpickle("cifar-10-batches-py/test_batch")
patcher = Patcher(16,32)
patches = patcher.get_patches_from_image_reduced(dicos['data'][0])
patcher.show_image(dicos['data'][0])
print(patches)
patcher.show_patches_reduced(patches[0])
#patcher.show_patches_reduced(normalized(patches[0]))
'''

