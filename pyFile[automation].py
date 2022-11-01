import numpy as np
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from pyFile import ARModel


modelAvl = ["vgg16","vgg19","inception","xception","resnet50","resnet101","densenet","inceptionResnet"]
modelSel = ["inception", "xception" ]
modelOnly = ["xception"]
losses = ["bce", "cce", "focal", "kld"]
lossSel = ["bce", "focal"]
lossOnly = ["focal"]

# normal or crossvalidation
exeState = "none"

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

def AR_normal(looper=1):
    model = ARModel(new=False)
    # Data Load
    (data, labels) = model.loadImages(r'/home/bishal/Research/Allergic-Rhinitis/Dataset/all/rotate', 
        plotType="R", shuffled=False, classification="multiclass", colorMode="RGB", cleanImageF=True, 
        resize=True, correctColor=False, contours=False, crop=True ,printImgDemo=False)

    # Data Preparation
    (data, labels) = model.prepareData(data, labels, weightedLossCalc=True)

    # Data Augmentation
    trainAug = model.setDataAugmentation(normalizeData=False, rotate=2, zoom=0.15, wShift=0.2, hShift=0.2, 
                                shear=0.15, hFlip=True, vFlip=False, generateImages=False)

    # Data Split
    (trainX, trainY, testX, testY) = model.setPartition(data, labels, testSize=0.20)

    for currentModel in modelSel:
        for loss in lossOnly:

            for i in range(looper):

                if looper>1:
                    # This part of code is for looper
                    # Read" new shuffled data 
                    print("\n Looper Number -> ", i+1)
                    print("\nLoading Shuffled Data")
                    (data, labels) = model.loadImages(r'/home/bishal/Research/Allergic-Rhinitis/Dataset/all/rotate', 
                                plotType="R", shuffled=True, classification="multiclass", colorMode="RGB", cleanImageF=True, 
                                resize=True, correctColor=False, contours=False, crop=True ,printImgDemo=False)
                                

                # Set Base Model
                currBModel = model.setBaseModel(currentModel)
                # Set Head Model Activation: (relu, leakyrelu, siren)
                currHModel = model.setHeadModel(currBModel, dropoutRate=0.5, activation="siren")
                # Set final Model
                finalModel = model.initModel(currBModel, currHModel, baseTrainable=False)
                # Set Hyperpaameters
                model.setHyperParameters(learningRate = 1e-3, epochs = 200, batchSize = 8)
                # Compile Model
                finalModel = model.compileModel(finalModel, loss=loss)
                # Start training
                (H, finalModel) = model.startTraining(finalModel, trainAug, trainX, trainY, 
                                                    testX, testY, weightedLoss=True, learningDecay=False, 
                                                    earlyStop=True, saveModel=False, verbose=0)
                # Start Testingboot
                predIdxs = model.startTesting(testX, testY, finalModel, voting=0)
                # Evaluate Model based on test outputs
                model.evalModel(predIdxs, testY)

                # Adds information to ARStats.csv File
                model.updateCSV()

                # Generate Plot - (ARSTATS - generates plot in Figures/ARStats.Plots)
                model.generatePlot(H, iterInfo="8", arstats=True)

                #model.getGradCams(type="test", model=finalModel, testX=testX)

def AR_crossValidation():
    model = ARModel()
    for currentModel in modelSel:
        model.crossValidate(path=r'/home/bishal/Research/Allergic-Rhinitis/Dataset/all/rotate', 
                    crop=True, clean=True, modelType=currentModel, loss="focal", activation="siren",
                    dropoutRate=0.5, batchSize=8, epochs=200, learningRate=1e-3, iter=7, 
                    baseTrainable=False, weightedLoss=True, learningDecay=False, earlyStop=True, voting=0, 
                    dataType = "CV-R", arstats=True)

def AR_leaveOneOut():

    model = ARModel()

    (data, labels) = model.loadImages(r'/home/bishal/Research/Allergic-Rhinitis/Dataset/all/rotate', 
        plotType="LOO-R", classification="multiclass", colorMode="RGB", cleanImageF=True, resize=True,
        correctColor=False, contours=False, crop=True ,printImgDemo=False)

    (data, labels) = model.prepareData(data, labels, weightedLossCalc=True)

    trainAug = model.setDataAugmentation(normalizeData=False, rotate=2, zoom=0.15, wShift=0.2, hShift=0.2, 
                                shear=0.15, hFlip=True, vFlip=False, generateImages=False)

    for currentModel in modelSel:
        for loss in lossOnly:

            currBModel = model.setBaseModel(currentModel)
            # Set Head Model Activation: (relu, leakyrelu, siren)
            currHModel = model.setHeadModel(currBModel, dropoutRate=0.5, activation="siren")
            finalModel = model.initModel(currBModel, currHModel, baseTrainable=False)
            model.setHyperParameters(learningRate = 1e-3, epochs = 200, batchSize = 8)
            finalModel = model.compileModel(finalModel, loss=loss)


            # for one V all
            y_hat = []
            y = []
            for i in range(len(data)):

                print("#### Iter - %s ####"%str(i+1))
                trainX = np.delete(data, [i], axis=0)
                trainY = np.delete(labels, [i], axis=0)
                testX = np.expand_dims(data[i], 0)
                testY = np.expand_dims(labels[i], 0)
                y.append(labels[i])
                print("Sizes : ", trainX.shape, "--", trainY.shape)
                print("Sizes : ", testX.shape, "--", testY.shape)
                
                (H, finalModel) = model.startTraining(finalModel, trainAug, trainX, trainY, 
                                            testX, testY, weightedLoss=True, learningDecay=False, earlyStop=True)
                                            
                                            # Start Testingboot
                predIdxs = model.startTesting(testX, testY, finalModel, voting=30)
                # Append to the Y hat list
                y_hat.append(predIdxs[0])
                
            model.eval2(y, y_hat)
            

            # Generate Plot - (ARSTATS - generates plot in Figures/ARStats.Plots)
            model.generatePlot(H, iterInfo="9", arstats=True, LOO=True)

            # Adds information to ARStats.csv File
            model.updateCSV()


if __name__ == "__main__":
    try:
        opType = sys.argv[1]
    except IndexError:
        print("Specify the operation type")
        exit()
    
    # fix GPU
    fix_gpu()   

    if opType == "normal":
        try:
            looper = int(sys.argv[2])
        except IndexError:
            looper = 1
        print("Looper set to ", looper)
        AR_normal(looper=looper)

    elif opType == "crossvalidation":
        AR_crossValidation()
    elif opType == "loo":
        AR_leaveOneOut()
    else:
        print("Operation does not exist :(")