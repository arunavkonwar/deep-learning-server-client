def addLocalLights(image, nbOfLights):
    import math
    result = np.copy(image)
    #nbOfLights = 3
    distanceXY = 350
    intensityVar = 0.20
    intensityNominal = 1.1
    spreadVar = 100
    spreadNominal = 500
    ambiantIllu = 0.4
    result = image*ambiantIllu # Setting the base result as black (no light situation)
    positionsX = [0]*nbOfLights
    positionsY = [0]*nbOfLights
    lightIntensities= [0]*nbOfLights
    lightSpreads = [0]*nbOfLights
    for i in range(0, nbOfLights):
        # Generating light source coordinates (as a distance from the center of the image for convenience)
        distanceX = uniform(-distanceXY, distanceXY)
        distanceY = uniform(-distanceXY, distanceXY)
        positionX = result.shape[0]/2 + distanceX
        positionY = result.shape[1]/2 + distanceY
        # The intensity decreases linearly between lightIntensity => 0, from the light position and max spread.
        lightIntensity = gauss(intensityNominal, intensityNominal*intensityVar)
        lightSpread = gauss(spreadNominal, spreadVar)
        
        positionsX[i] = positionX
        positionsY[i] = positionY
        lightIntensities[i] = lightIntensity
        lightSpreads[i] = lightSpread
        
        #print "light source = ", distanceX, "/", distanceY, " => Intensity = ", lightIntensity, " || Spread = ", lightSpread
    
    for i in range(0, nbOfLights):
        distX = np.arange(image.shape[0])
        distX = (distX - positionsX[0])*(distX - positionsX[0])
        distX = np.repeat(distX, image.shape[1], axis=0)
        distX = np.reshape(distX, [image.shape[0], image.shape[1]])
        #print "distX Shape = ", distX.shape
        #print distX
        
        distY = np.arange(image.shape[1])
        distY = (distY - positionsY[0])*(distY - positionsY[0])
        distY = np.repeat(distY, image.shape[0], axis=0)
        distY = np.transpose(np.reshape(distY, [image.shape[1], image.shape[0]]))
        #print "distY Shape = ", distY.shape
        #print distY
    
        distXY = np.sqrt(distX + distY)
        #print "distXY Shape = ", distXY.shape
        #print distXY
    
        lightEffect = (lightSpreads[i] - distXY)/lightSpreads[i]
        lightEffect = np.clip(lightEffect, 0, 10) # Restrincting the effect to a positive value
        #print lightEffect
        lightEffect = np.repeat(lightEffect, 3, axis = 1)
        lightEffect = np.reshape(lightEffect, [image.shape[0], image.shape[1], 3])
        result = result + np.multiply(image, lightEffect*lightIntensities[i]/nbOfLights)
    result = np.clip(result, 0, 1)
    
    #for x in range(0, image.shape[0]):
    #    for y in range(0, image.shape[1]):
    #        for i in range(0, nbOfLights):
    #            distanceFromPoint = math.sqrt((x-positionsX[i])*(x-positionsX[i]) + (y-positionsY[i])*(y-positionsY[i]))
    #            #print "DistFromPoint = ", distanceFromPoint
    #            lightEffectAtPoint = (lightSpreads[i] - distanceFromPoint)/lightSpreads[i] # Getting the % of effect at point, depending of the distance of the source
    #            if lightEffectAtPoint <= 0 :
    #                lightEffectAtPoint=0
    #            #print "Light Effect at point = ", lightEffectAtPoint
    #            result[x, y, :] = result[x, y, :] + image[x, y, :] * lightIntensities[i] * lightEffectAtPoint/nbOfLights # Adding the effect of the light, based on the original value of the pixel and the strength of the light affecting this pixel
    #            if result[x, y, 0] < 0:
    #                result[x, y, :] = 0
    #            if result[x, y, 0] > 1:
    #                result[x, y, :] = 1
                
    #result = result
        
    return result


def insertRandomOcclusion(image, nbOfOcclusions):
    import random, os
    import skimage.segmentation
    import time
    import matplotlib.pyplot as plt
    import cv2
    #print 'RUNNING INSERTION'
    #raise Exception('test Exception')
    result = np.copy(image)
    # Select a random image
    directory = '/local/qbateux/All/'
    #directory = '/home/quentin/Downloads/LabelMe-12-50k/All/'

    for i in range(0, nbOfOcclusions):
    
        random_filename = random.choice([
            x for x in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, x))
        ])

        #img = caffe.io.load_image(directory+random_filename)
        img = cv2.imread(directory+random_filename)

        #print("Selected File = " + random_filename)
        #plt.imshow(img)
        #plt.show()
    
        # Compute the super-pixels of the selected image
        superpixel = skimage.segmentation.slic(img, n_segments=10, compactness=20, sigma=1)

        #plt.imshow(superpixel%7, cmap='Set2')
        #plt.show()
    
        sPixOk = False
        count = 0
        while not sPixOk:
            # Select one of the super-pixels
            segIndex = int(random.uniform(0, max(np.unique(superpixel))))
            mask = np.zeros(img.shape[:2], dtype = "uint8")
            mask[superpixel == segIndex] = 255
            #cv2.imshow("Mask", mask)
            maskedImage = cv2.bitwise_and(img, img, mask = mask)
            
            #plt.imshow(maskedImage)
            #plt.show()
        
            ret,thresh = cv2.threshold(cv2.cvtColor(maskedImage,cv2.COLOR_BGR2GRAY),0,255,0)
            contours,hierarchy = cv2.findContours(thresh, 1, 2)
            #print len(contours)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            if(w > 20 or h > 20 or (count >= 10 and h>0 and w>0)):
                sPixOk = True
            else:
                count = count+1
            #cv2.rectangle(maskedImage,(x,y),(x+w,y+h),(0,255,0),2)
            #plt.imshow(maskedImage)
            #plt.show()
            ##
        superPixelImage = maskedImage[y:y+h, x:x+w]
        #print superPixelImage.shape
        #print superPixelImage[0, :, :]
        #superPixelImage[:, :, 0] = (superPixelImage[:, :, 0]+superPixelImage[:, :, 1]+superPixelImage[:, :, 2])/3
        superPixelImage[:, :, 0] = 0.299*superPixelImage[:, : , 0] + 0.587*superPixelImage[:, : , 1] + 0.114*superPixelImage[:, : , 2]
        superPixelImage[:, :, 1] = superPixelImage[:, :, 0]
        superPixelImage[:, :, 2] = superPixelImage[:, :, 0]
        #print h, "//", w
        # Insert the super-pixel in a random spot of the base image
        #posX = int(uniform(-h, image.shape[0]-h-1))
        #posY= int(uniform(-w, image.shape[1]-w-1))
        posX = int(uniform(0, 239)) # Hard Coded limit based on the smallest dimension of the image. TODO: Fix the dimension overflow !!!
        posY= int(uniform(0, 239))
        #print image.shape[1]-h-1, "/", posX+h
        #print image.shape[1], "/",  image.shape[1]-h
        #print "POS = ",posX, "/", posY
        #print "Image = ", result.shape
        #print "SuperPix = ", superPixelImage.shape
        #print "w,h =", w, "/", h
        #print "x = ", posX+h, "/", superPixelImage.shape[0]
        #print "y = ", posY+w, "/", superPixelImage.shape[1]
        # WARNING : superPix X and Y and reversed !
        # w => y ; h => x
        #plt.imshow(superPixelImage)
        #plt.show()
        
        for i in range(0, h-1):
            for j in range(0, w-1):
                if superPixelImage[i, j].all() != 0 and i+posX < result.shape[0] and j+posY < result.shape[1]:
                    result[i+posX,j+posY] = superPixelImage[i, j]/255.0
    
        #for i in range(posX, posX+h):
        #    for j in range(posY, posY+w):
        #        if superPixelImage[i-posX, j-posY].all() != 0:
        #            result[i,j] = superPixelImage[i-posX, j-posY]/255.0
                
        # Return the modified image
        #time.sleep(1000)
    #plt.imshow(result)
    #plt.show()
    return result
    

def modifyImage(image, method, modificationParameter=0, posChangeX=0, posChangeY=0, nbOfLights=0, nbOfOcclusions=0):
    from scipy import ndimage
    #print 'Modif = ' + `method` + '_' + `modificationParameter`
    result = np.copy(image)
    if method == 0 : # Illumination change. Expected x E [0, N]. where x is the % of illumination change, 1 = no change, 0 = black
        modificationParameter = 1 - modificationParameter
        result =  np.clip(result* modificationParameter, 0, 1)
    if method == 1 : # Addition of blur (gaussian filter)
        result = ndimage.gaussian_filter(result, sigma=modificationParameter) # parameter is int [0, 10]
    if method == 2 : # Addition of white noise, parameter is float [0, 1]
        result = np.clip(result + modificationParameter * result.std() * np.random.random(result.shape), 0, 1)
    if method == 3: # Addition of an occlusion square in the middle of the image. Parameter is the radius (in pixels)
        result[result.shape[0]/2+posChangeX-modificationParameter:result.shape[0]/2+posChangeX+modificationParameter, result.shape[1]/2+posChangeY-modificationParameter:result.shape[1]/2+posChangeY+modificationParameter, :] = 255
    if method == 4: # Creation of N local light sources, in order to modify the illumination locally, instead of globaly only
        result = addLocalLights(image, nbOfLights)
    if method == 5: # Insertion of N occlusions, chosen as a piece extracted randomly from a random image
        result = insertRandomOcclusion(image, nbOfOcclusions)
    return result

if __name__ == '__main__':
    caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    import sys
    sys.path.insert(0, caffe_root + 'python')
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (10, 10)        # large images
    plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleadi
    import h5py, os
    import caffe
    import numpy as np
    from PIL import Image
    from shutil import copyfile
    from shutil import rmtree
    from random import gauss, uniform
    import scipy.misc
    
    #datasetPath = '/media/quentin/OSDisk/testLocalLights'
    #datasetPath = '/local/qbateux/Afma_Hollywood_NoDiff'
    #datasetPath = '/local/qbateux/Afma_Hollywood_NoDiff_Finer'
    #datasetPath = '/local/qbateux/Afma_Hollywood_NoDiff_Finer_BIG'
    #datasetPath = '/media/quentin/OSDisk/testLocalLights'
    datasetPath = '/local/qbateux/Afma_Hollywood_NoDiff_Finer_MEDIAN'
    datasetPath = '/local/qbateux/Afma_Hollywood_NoDiff_Finer_SMALL'
    datasetPath = '/local/qbateux/Afma_Castle_MEDIAN'
    datasetPath = '/local/qbateux/Afma_Castle_MEDIAN2'
    datasetPath = '/local/qbateux/RealVS_test1'
    copyPrefix = '_degraded_occlusions/'
    # Prob = chance that the current image has to be affected by this particular degradation (multiple effects can be stacked)
    # Var = variance of the intensity of the considered degradation
    illuProb = 0 
    illuVar = 0.2 # gaussian noise ; final values extrema= [min 0, max 1]
    illuMin = 0
    illuMax = 1

    blurProb = 0
    blurVar =  3 # gaussian noise ; final values extrema= [min 0, max 10]
    blurMin = 0
    blurMax = 10

    noiseProb = 0
    noiseVar = 0.5 # gaussian noise ; final values extrema= [min 0, max 1]
    noiseMin = 0
    noiseMax = 1

    occlusionProb = 0
    occlusionSizeVar = 20 # gaussian noise [1, 227/2]
    occlusionSizeMin = 0
    occlusionSizeMax = 113
    
    localLightsProb = 1.0
    localLightsNb = 2 # Nb of lights that will be used

    occlusionAdvancedProb = 1.0
    occlusionAdvancedNb = 4

    display = False
    useDualImages = True #False for PoseNet-style, True for FlowNet-style
    
    continueFromBefore = False
    restartFromNPoints = 10 # We restart 10 points before the last one written
    
    indexFile = datasetPath+'/DatasetIndex'
    degradedDatasetPath = datasetPath+copyPrefix
    
    if not continueFromBefore: # If we start from scratch, we want to erase all possible previous results before starting the write the new ones.
        if os.path.exists(degradedDatasetPath):
            rmtree(degradedDatasetPath)
        if not os.path.exists(degradedDatasetPath):
            os.makedirs(degradedDatasetPath)
    print indexFile, ' // ', degradedDatasetPath+'DatasetIndex'
    copyfile(indexFile, degradedDatasetPath+'DatasetIndex')
    #print TEST
    with open( indexFile, 'r' ) as T :
        lines = T.readlines()
        #print 'lines shape =', len(lines)
        if continueFromBefore:
            with open( datasetPath+copyPrefix+'DatasetIndex', 'r' ) as T :
                linesFromLastSession = T.readlines()
                nbOfOldLines = len(linesFromLastSession)
                iOld = nbOfOldLines-restartFromNPoints# We restart i at the last valid index from the last session
                print 'Starting from last session, at image i = ', iOld 
            # We cut the line list, so that we skip the already seen ones.
            lines = lines[iOld:len(lines)-1]
        else :
            iOld = 0
    
    for i,l in enumerate(lines):
        if(i % 1 == 0):
            print 'Image = ', i+iOld
        l = l.strip()
        sp = l.split(';') # Take out the extra whitespaces, newlines...etc
        if not useDualImages:
            img = caffe.io.load_image( sp[0] )
        else:
            img = caffe.io.load_image( sp[0] + '_1.png')
            img2 = caffe.io.load_image( sp[0] + '_2.png')
        #print 'Image type = ', type(img)
        #plt.figure()
        #plt.imshow(img)
        #plt.show()
        
        ## Degrading the image
        # For each degradation, we test is it will be applied or not
        if uniform(0, 1) <  illuProb :
            #print 'Changing illu'
            param = gauss(0, illuVar)
            if(param < illuMin) :
                param = illuMin
            if(param > illuMax) :
                param = illuMax
            #print 'Param = ', param
            img = modifyImage(img, 0, param)
            if useDualImages:
                img2 = modifyImage(img2, 0, param)
        
        if uniform(0, 1) <  blurProb :
            #print 'Adding blur'
            param = gauss(0, blurVar)
            if(param < blurMin) :
                param = blurMin
            if(param > blurMax) :
                param = blurMax
            #print 'Param = ', param
            img = modifyImage(img, 1, param)
            if useDualImages:
                img2 = modifyImage(img2, 1, param)

        if uniform(0, 1) <  noiseProb :
            #print 'Adding white noise'
            param = gauss(0, noiseVar)
            if(param < noiseMin) :
                param = noiseMin
            if(param > noiseMax) :
                param = noiseMax
            #print 'Param = ', param
            img = modifyImage(img, 2, param)
            if useDualImages:
                img2 = modifyImage(img2, 2, param)

        if uniform(0, 1) <  occlusionProb :
            #print 'Adding Occlusion'
            param = gauss(0, occlusionSizeVar)
            if(param < occlusionSizeMin) :
                param = occlusionSizeMin
            if(param > occlusionSizeMax) :
                param = occlusionSizeMax
            posX = uniform(-110, 110-param)
            posY = uniform(-110, 110-param)
            #print 'Param = ', param
            img = modifyImage(img, 3, param, posChangeX=posX, posChangeY=posY)
            if useDualImages:
                img2 = modifyImage(img2, 3, posChangeX=posX, posChangeY=posY)
            
        if uniform(0, 1) <  localLightsProb :
            #print 'Adding Occlusion'
            #param = gauss(0, occlusionSizeVar)
            #if(param < occlusionSizeMin) :
            #    param = occlusionSizeMin
            #if(param > occlusionSizeMax) :
            #    param = occlusionSizeMax
            #posX = uniform(-110, 110-param)
            #posY = uniform(-110, 110-param)
            #print 'Param = ', param
            img = modifyImage(img, 4, nbOfLights=localLightsNb)
            if useDualImages:
                 img2 = modifyImage(img2, 4, nbOfLights=localLightsNb)

        if uniform(0, 1) <  occlusionAdvancedProb :
            #print 'Adding Occlusion'
            #param = gauss(0, occlusionSizeVar)
            #if(param < occlusionSizeMin) :
            #    param = occlusionSizeMin
            #if(param > occlusionSizeMax) :
            #    param = occlusionSizeMax
            #posX = uniform(-110, 110-param)
            #posY = uniform(-110, 110-param)
            #print 'Param = ', param
            try:
                img = modifyImage(img, 5, nbOfOcclusions=occlusionAdvancedNb)
                if useDualImages:
                     img2 = modifyImage(img2, 5, nbOfOcclusions=occlusionAdvancedNb)
            except: # Re-try the occlusion insertion
                print 'Trying again'
                img = modifyImage(img, 5, nbOfOcclusions=occlusionAdvancedNb)
                if useDualImages:
                     img2 = modifyImage(img2, 5, nbOfOcclusions=occlusionAdvancedNb)
                     
        if display:    
            plt.figure()
            plt.imshow(img)
            plt.show()

        # Saving the image in the degraded Dataset Folder
        if not useDualImages:
            scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(degradedDatasetPath+'im_' + '{:07.0f}'.format(i+iOld)  + '.png')
        else:
            scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(degradedDatasetPath+'im_' + '{:07.0f}'.format(i+iOld)  + '_1.png')
            scipy.misc.toimage(img2, cmin=0.0, cmax=1.0).save(degradedDatasetPath+'im_' + '{:07.0f}'.format(i+iOld)  + '_2.png')
        #scipy.misc.toimage(img).save(degradedDatasetPath+'im_' + '{:07.0f}'.format(i)  + '.png')
        
        # Modifying the paths in the new 'DatasetIndex'
        text_file = open(degradedDatasetPath+'/DatasetIndex', "a")
        if not useDualImages:
            str = degradedDatasetPath+'im_' + '{:07.0f}'.format(i+iOld)  + '.png'
        else:
            str = degradedDatasetPath+'im_' + '{:07.0f}'.format(i+iOld)
        for i in range(1, len(sp)):
            str = str + ';'+sp[i]
        text_file.write(str+'\n')
        text_file.close()
