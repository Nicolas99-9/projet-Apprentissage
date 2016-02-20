import numpy as np
import cPickle
import matplotlib.pyplot as plt


class Patcher:
 
    def __init__(self,size_patch,imageSize):
        self.imageSize = imageSize
        self.sizeSquare = size_patch
        self.number_patches = (imageSize/size_patch)

    def show_informations(self):
        print("Informations about the patcher : ")
        print("Image size : ",self.imageSize)
        print("Size square :",self.sizeSquare)
        print("Number of patches per line : ", self.number_patches)
        print("Number of patches in the image : ",  self.number_patches*self.number_patches) 

    def get_patches_from_image(self,image):
        #a[:,1] get the second colum
        after = np.array(image).reshape((self.imageSize,self.imageSize*3),order='F')
        print(after.shape)
        patches= []
        for i in range(self.number_patches):
            for j in xrange(0,self.number_patches):
                tmp = after[i*self.sizeSquare:i*self.sizeSquare+self.sizeSquare,j*self.sizeSquare:j*self.sizeSquare+self.sizeSquare*3]
                patches.append(tmp.reshape(self.sizeSquare*self.sizeSquare*3))
        #print("after : ",after)
        #print("--------------------------------------")
        #print(patches)
        return list(patches)

    def show_patches(self,patches):
        #print("l1",len(patches),len(patches[0]))
        tmp = np.array(patches)
        print("debugf ",tmp.shape)
        tmp = tmp/255.0 
        to_display = []
        for i in range(len(patches)):
            to_add = []
            for j in xrange(0,(self.sizeSquare)*3,3):
                to_add.append(patches[i][j:j+3])
            to_display.append(to_add)
        to_display = np.array(to_display)
        plt.imshow(to_display)
        plt.show()
        
#test

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

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

dicos = unpickle("cifar-10-batches-py/data_batch_1")
patcher = Patcher(16,32)
patcher.show_informations()
patches = patcher.get_patches_from_image(dicos['data'][0])
print("shape des patches,",np.array(patches).shape)
print(patches[0])
show_image(dicos['data'][0])
patcher.show_patches(patches[0])
