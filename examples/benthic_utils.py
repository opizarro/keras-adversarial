import os
import cv2
import numpy as np

# benthoz images location
image_dir = "/Volumes/Samsung_T3/Benthoz2015/"
# images stored as png



def benthic_process(x):
    x = x.astype(np.float32) / 255.0
    return x

def benthoz_data():
    # images in set
    n = 9600
    # fraction to be 'test'
    testfrac = 0.2
    rescale_im = 0.125
    #
    ntest = int(n * testfrac)
    ntrain = n - ntest

    imsize = 32

    #xtrain = np.empty((ntrain, 128, 170, 3), dtype='uint8')
    #xtest = np.empty((ntest, 128, 170, 3), dtype='uint8')

    xtrain = np.zeros((ntrain, imsize, imsize, 3))
    xtest = np.zeros((ntest, imsize, imsize, 3))


    i = 0
    #loop through images
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            #read
            print("processing image number {} name {} ".format(i,filename))
            image = cv2.imread(os.path.join(image_dir,filename))
            #small = cv2.resize(image, (0,0), fx=rescale_im,fy=rescale_im)
            small = cv2.resize(image, (imsize,imsize))
            #print("mean small image {}".format(np.mean(small)))
            #cv2.imwrite("output/aae-benthic/" + filename, small)
            #print("counter i {}".format(i))
            if i < ntrain:
                # save a train array
                #print("i {} less than num train samps {}".format(i,ntrain))
                xtrain[i,:,:,:] = benthic_process(small)
            elif i >= ntrain and i < n:
                #print("i {} equal or greater than {} and less than {}".format(i,ntrain,n))
                xtest[i-ntrain,:,:,:] = benthic_process(small)
            else:
                #print("mean xtrain {}".format(np.mean(xtrain)))
                break
            i += 1

    return xtrain, xtest