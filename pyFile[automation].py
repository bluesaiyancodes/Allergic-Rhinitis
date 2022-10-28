from pyFile import ARModel
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np


modelAvl = ["vgg16","vgg19","inception","xception","resnet50","resnet101","densenet","inceptionResnet"]
modelSel = ["inception", "xception" ]
modelOnly = ["xception"]
losses = ["bce", "cce", "focal", "kld"]
lossSel = ["bce", "focal"]
lossOnly = ["focal"]

# normal or crossvalidation
exeState = "oneVall"

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()

if exeState == "normal":
    model = ARModel(new=False)
    # Data Load
    (data, labels) = model.loadImages(r'/home/bishal/Research/Allergic-Rhinitis/Dataset/all/rotate', 
        plotType="R", classification="multiclass", colorMode="RGB", cleanImageF=True, resize=True,
        correctColor=False, contours=False, crop=True ,printImgDemo=False)

    # Data Preparation
    (data, labels) = model.prepareData(data, labels, weightedLossCalc=True)

    # Data Augmentation
    trainAug = model.setDataAugmentation(normalizeData=False, rotate=2, zoom=0.15, wShift=0.2, hShift=0.2, 
                                shear=0.15, hFlip=True, vFlip=False, generateImages=False)

    # Data Split
    (trainX, trainY, testX, testY) = model.setPartition(data, labels, testSize=0.20)

    for currentModel in modelSel:
        for loss in lossOnly:
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
                                                    testX, testY, weightedLoss=True, learningDecay=False, earlyStop=True)
            # Start Testingboot
            predIdxs = model.startTesting(testX, testY, finalModel, voting=0)
            # Evaluate Model based on test outputs
            model.evalModel(predIdxs, testY)

            # Adds information to ARStats.csv File
            model.updateCSV()

            # Generate Plot - (ARSTATS - generates plot in Figures/ARStats.Plots)
            model.generatePlot(H, iterInfo="8", arstats=True)

            #model.getGradCams(type="test", model=finalModel, testX=testX)
            

if exeState == "crossvalidation":
    model = ARModel()
    for currentModel in modelSel:
        model.crossValidate(path=r'/home/bishal/Research/Allergic-Rhinitis/Dataset/all/rotate', 
                    crop=True, clean=True, modelType=currentModel, loss="focal", activation="siren",
                    dropoutRate=0.5, batchSize=8, epochs=200, learningRate=1e-3, iter=7, 
                    baseTrainable=False, weightedLoss=True, learningDecay=False, earlyStop=True, voting=0, 
                    dataType = "CV-R", arstats=True)


if exeState == "oneVall":
    model = ARModel()

    (data, labels) = model.loadImages(r'/home/bishal/Research/Allergic-Rhinitis/Dataset/all/rotate', 
        plotType="R", classification="multiclass", colorMode="RGB", cleanImageF=True, resize=True,
        correctColor=False, contours=False, crop=True ,printImgDemo=False)
    (data, labels) = model.prepareData(data, labels, weightedLossCalc=True)
    trainAug = model.setDataAugmentation(normalizeData=False, rotate=2, zoom=0.15, wShift=0.2, hShift=0.2, 
                                shear=0.15, hFlip=True, vFlip=False, generateImages=False)

    for currentModel in modelSel:
        for loss in lossOnly:
            # for one V all
            y_hat = []
            y = []
            for i in range(len(data)):
                trainX = np.delete(data, [i], axis=0)
                trainY = np.delete(labels, [i], axis=0)
                testY = np.expand_dims(data[i], 0)
                testX = np.expand_dims(labels[i], 0)
                y.append(labels[i])

                print("Sizes : ", trainX.shape, "--", trainY.shape)
                
                currBModel = model.setBaseModel(currentModel)
                # Set Head Model Activation: (relu, leakyrelu, siren)
                currHModel = model.setHeadModel(currBModel, dropoutRate=0.5, activation="siren")
                finalModel = model.initModel(currBModel, currHModel, baseTrainable=False)
                model.setHyperParameters(learningRate = 1e-3, epochs = 200, batchSize = 8)
                finalModel = model.compileModel(finalModel, loss=loss)
                (H, finalModel) = model.startTraining(finalModel, trainAug, trainX, trainY, 
                                                    testX, testY, weightedLoss=True, learningDecay=False, earlyStop=True)
                # Start Testingboot
                predIdxs = model.startTesting(testX, testY, finalModel, voting=0)
                # Append to the Y hat list
                y_hat.append(predIdxs)
                
                model.eval2(y, y_hat)
                # Adds information to ARStats.csv File
                model.updateCSV()

                # Generate Plot - (ARSTATS - generates plot in Figures/ARStats.Plots)
                #model.generatePlot(H, iterInfo="8", arstats=True)

                #model.getGradCams(type="test", model=finalModel, testX=testX)