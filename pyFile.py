from imutils import paths
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import random
import timeit
from datetime import datetime

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, InceptionResNetV2, DenseNet121, ResNet50V2, ResNet101V2, Xception, InceptionV3
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_addons as tfa
from tf_siren import SinusodialRepresentationDense

from extraMethods.ImageCorrection import colorCorrect
from extraMethods.gradCam import  getHeatMap, saveGradCam
from extraMethods.imagePreProcessing import cropImage, addContours, resizeImage, cleanImage, savePlot


class ARModel:
    def __init__(self, new=False):
        # Initialize data and labels
        # 데이터 및 레이블 초기화
        self.meta = {}
        # Set Seeding for Tensorflow operations
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # New FileOPerations
        if new:
            self.fileOP = open("ARStats.csv", "w")
            self.fileOP.write("imageid,imageType,model,loss,classType,colorMode,clean,crop,resize,colorCorrect,baseTrain,activation,wloss,lr,learningDecay,earlyStop,epochsMax,epochs,voting, trainTime, accuracy\n")
            self.fileOP.close()
            self.meta["imgNumber"] = 0
        else:
            # Retrieve Img number to save image with the corresponding ARStats.csv entry
            with open('ARStats.csv', 'r') as f:
                lastLine = f.readlines()[-1].split(",")[0]
                imNumber = int(lastLine) + 1 
                self.meta["imgNumber"] = imNumber

    def loadImages(self, path=r'C:\Users\cvpr\Documents\Bishal\Allergic Rhinitis\Dataset\rotate', plotType="all", 
                    classification="multiclass", colorMode = "RGB", cleanImageF=False, resize=False,
                    crop = False, correctColor=False, contours=False, printImgDemo=False):
        print("[INFO]: Trying to Read the images from ", path)
        data = []
        labels = []
        #  Configure the Image Location            
        # 이미지 위치 구성하기
        self.imagePaths =  list(paths.list_images(path))
        # Plot type is used only in title of plot image
        # Adding to metadata
        self.meta["dataInfo"] = plotType
        self.meta["classification"] = classification

        singleImagePrintLabel = []

        # Save meta information for future img manipulation


        # Shuffle the items in the image data paths
        #shuffle(self.imagePaths)
        
        
        # Formatting data and labels
        for imagePath in self.imagePaths:
            # Extract the class label from file name and append to labels
            # 파일 이름에서 클래스 레이블을 추출하고 레이블에 추가함
            label = imagePath.split(os.path.sep)[-2]

            # dividing labels based on multiclass or binary
            if classification=="binary":
                if label=="2":
                    labels.append("1")
                else:
                    labels.append(label)
            else:
                labels.append(label)
        

            # Load the image, swap color channels, and resize it to be a fixed 224x224 pixels while ignoring the aspect ratio
            # 이미지를 로드하고, 컬러 채널을 스왑하고, 가로 세로 비율을 무시하고 고정 224x224 픽셀로 크기를 조정함
            if colorMode=="LAB":
                image = cv2.imread(imagePath)
                image = image.astype("float32")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            else:
                image = cv2.imread(imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To clean the image that is remove the black noise from the picture
            if cleanImageF:
                image = cleanImage(image)

            # To crop the image to the subject size
            if crop:
                image = cropImage(image)
                # Cropping requires re-size due to irregularity in the  dim of image cropping
                resize = True
                image = resizeImage(image, (224,224))

            # If resize is requested, check if it is not alreaddy resized by crop
            if resize:
                if len(image) != 224:
                    image = resizeImage(image, (224, 224))
            # If Color Correction is required
            # Perform Color Correction
            if correctColor:
                image = colorCorrect(image)

            # Contours is added in image if required
            if contours:
                image = addContours(image)
        
            # Setting code to print image - one from each class
            if label not in singleImagePrintLabel:
                singleImagePrintLabel.append(label)
                if printImgDemo:
                    savePlot(image, label)

            # Append to data
            # 데이터에 추가
            data.append(image)
            # Adding to metadata
            self.meta["imageCount"] = len(data)
            self.meta["imageType"] = plotType
            self.meta["classType"] = classification
            self.meta["colorMode"] = colorMode
            self.meta["clean"] = cleanImageF
            self.meta["crop"] = crop
            self.meta["resize"] = resize
            self.meta["colorCorrect"] = correctColor
        print("Images found :", len(data))
        return (data, labels)

    def prepareData(self, data, labels, weightedLossCalc=False):

        # Convert the data and labels to NumPy arrays while scaling the pixel intensities to the range [0,1]
        # 픽셀 강도를 [0,1] 범위로 조정하면서 데이터와 레이블을 NumPy 배열로 변환

        print("[INFO]: Preparing Data")

        # 
        #data = np.array(data)
        # Zero Mean Normalization 
        #data = (data - data.mean()) / data.var()
        data = np.array(data) / 255.0
        labels = np.array(labels)

        # Store Image Dimentions
        self.meta["imgDim"] = data[0].shape
        print(self.meta)

        if(weightedLossCalc):
            self.class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
            self.meta["weightedLoss"] = self.class_weights
        # Perform the one-hot encoding on the labels
        # 레이블에 대해 원핫 인코딩 수행
        self.lb = LabelBinarizer()
        labels = self.lb.fit_transform(labels)

        # Categorize data if classification mode is binary
        if self.meta["classification"]=="binary":
            labels = to_categorical(labels)
        
        return (data, labels)

    def setDataAugmentation(self, normalizeData=False, rotate=2, zoom=[0.5, 1.0], wShift=0.2, hShift=0.2, shear=0.15, hFlip=True, vFlip=False, generateImages=False):
        # Initialize the training data augmentation
        # 교육 데이터 억멘테이션 초기화

        # the brightness augmentation was removed as it lead to poor training.
        if normalizeData:
            trainAug = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=rotate, 
                            zoom_range=zoom, width_shift_range=wShift, height_shift_range=hShift,
		                    shear_range=shear, fill_mode="constant", horizontal_flip=hFlip, vertical_flip=vFlip)
        else:
            trainAug = ImageDataGenerator(rotation_range=rotate, zoom_range=zoom, width_shift_range=wShift, height_shift_range=hShift,
		                     shear_range=shear, fill_mode="constant", horizontal_flip=hFlip, vertical_flip=vFlip)
         # Adding to metadata
        self.meta["dataAugmentation"] = {}
        self.meta["dataAugmentation"]["rotate"] = rotate
        self.meta["dataAugmentation"]["zoom"] = zoom
        self.meta["dataAugmentation"]["wShift"] = wShift
        self.meta["dataAugmentation"]["hShift"] = hShift
        self.meta["dataAugmentation"]["shear"] = shear
        self.meta["dataAugmentation"]["hFlip"] = hFlip
        self.meta["dataAugmentation"]["vFlip"] = vFlip

        print("[INFO]: Augmenting Data with - ")
        print(self.meta["dataAugmentation"])
        return trainAug
   
    def setPartition(self, data, labels, testSize=0.20):
        # Partition the data into training and testing splits using 80% of the training data and the remaining 20% for testing
        # 교육 데이터의 80%, 테스트에 20%를 사용하여 데이터를 교육 및 테스트 분할로 분할
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=testSize, stratify=labels, random_state=19)
        # Adding to metadata
        self.meta["partitionRatio"] = str(int((1-testSize)*100))+ ":" +str(int(testSize*100))
        print("[INFO]: Patition Set to - ", self.meta["partitionRatio"])
        return (trainX, trainY, testX, testY)

    def setBaseModel(self, modelType="vgg16"):
        # Load the model network, ensuring the Head-FC layer sets are left off
        # Head-FC 레이어 세트가 포함되지 않도록 VGG16 네트워크를 로드한다

        shape = ()
        if modelType == "vgg16":
            baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=self.meta["imgDim"]))
        
        elif modelType == "vgg19":
            baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=self.meta["imgDim"]))
        
        elif modelType == "inception":
            baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=self.meta["imgDim"]))
        
        elif modelType == "xception":
            baseModel = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=self.meta["imgDim"]))

        elif modelType == "resnet50":
            baseModel = ResNet50V2(weights="imagenet", include_top=False, input_tensor=Input(shape=self.meta["imgDim"]))

        elif modelType == "resnet101":
            baseModel = ResNet101V2(weights="imagenet", include_top=False, input_tensor=Input(shape=self.meta["imgDim"]))
        
        elif modelType == "densenet":
            baseModel = DenseNet121(weights="imagenet", include_top=False, input_tensor=Input(shape=self.meta["imgDim"]))

        elif modelType == "inceptionResnet":
            baseModel = InceptionResNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=self.meta["imgDim"]))
        
        else:
            print("Model not found. Proceding with VGG16.")
            modelType = "vgg16"
            self.baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
        # Adding to metadata
        self.meta["model"] = modelType
        print("[INFO]: Model Selected - ", modelType)

        return baseModel

    def setHeadModel(self, baseModel, dropoutRate = 0.5, activation="relu"):
        # Construct the head model that will be placed on the top of the base model
        # 보디 모델의 맨 위에 배치할 헤드 모델 구성
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4,4))(headModel)
        headModel = Flatten(name="flatten")(headModel)

        if activation=="relu":
            headModel = Dense(64, activation="relu")(headModel)
        elif activation=="leakyrelu":
            headModel = Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(headModel)
        elif activation=="siren":
            headModel = SinusodialRepresentationDense(64, activation='sine', w0=1.0)(headModel)
            
        headModel = Dropout(dropoutRate)(headModel)
        # Head Model Configuration based on classfication type
        if self.meta["classification"]=="binary":
            headModel = Dense(2, activation="softmax")(headModel) 
        else:  
            headModel = Dense(3, activation="softmax")(headModel)
        # Adding to metadata
        self.meta["dropoutRate"] = dropoutRate
        self.meta["activation"] = activation
        return headModel
    
    def initModel(self, baseModel, headModel, baseTrainable=False):
        # Place the Head-FC model on top of the Base model - This become the actual model that we will train
        # Head-FC 모델을 보디 모델 위에 배치한다. 이것이 우리가 교육할 실제 모델이 될 것이다.
        model = Model(inputs=baseModel.input, outputs=headModel)

        # Make sure that the basemodel layers will not be trained and only head model will be trained.
        # 보디 모델 레이어가 훈련되지 않고 헤드 모델만 훈련되는지 확인한다.
        for layer in baseModel.layers:
            layer.trainable = baseTrainable
        print("[INFO]: Initializing Model")
        # Adding meta information
        self.meta["baseTrain"] = baseTrainable
        return model

    def setHyperParameters(self, learningRate = 1e-3, epochs = 100, batchSize = 8):
        # Set the hyper-parameters
        # 하이퍼 파라미터 설정
        # INIT_LR = 1e-3
        self.INIT_LR = learningRate
        self.EPOCHS = epochs
        self.BS = batchSize
        # Adding to metadata
        self.meta["learningRate"] = learningRate
        self.meta["epochsMax"] = epochs
        self.meta["batchSize"] = batchSize
        
        print("[INFO]: Hyperparameters Set")

    def compileModel(self, model, loss="bce"):
        # Compile the Model
        # 모델 컴파일
        print("[INFO]: Compiling Model")
        opt = Adam(learning_rate=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)

        if loss=="cce":
            model.compile(loss=losses.CategoricalCrossentropy(), optimizer=opt, metrics=["accuracy"])
        elif loss=="bce":
            model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        elif loss=="focal":
            model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(), optimizer=opt, metrics=["accuracy"])
        elif loss=="scce":
            model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        elif loss=="kld":
            model.compile(loss="kullback_leibler_divergence", optimizer=opt, metrics=["accuracy"])
        else:
            print(loss+" not available. Proceeding with binary_crossentropy")
            loss=bce
            model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])  
        # adding meta information
        self.meta["loss"] = loss 
        return model

    def startTraining(self, model, trainAug, trainX, trainY, testX, testY, weightedLoss=False, learningDecay= False, earlyStop=False):                                                        
        # Train the Network Model
        # 모델 교육
        print("[INFO] Model Training")
        model_checkpoint = ModelCheckpoint('AR_MODEL.hdf5', monitor='loss',verbose=1, save_best_only=True)

        # Gradual Learning Rate reduction
        learningRateReduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=1e-8)
        # Early Stopping the traininga nd storing the best accuracy
        earlyStopping = EarlyStopping(
                                    monitor='val_accuracy', 
                                    patience=95,
                                    restore_best_weights=True)

        # Setting Callbacks
        callback = [model_checkpoint]
        if learningDecay:
            callback.append(learningRateReduction)
        if earlyStop:
            callback.append(earlyStopping)
        
        # Add to Meta
        self.meta["callbacks"] = callback
        
        # Start Timer
        start = timeit.default_timer()
        if(weightedLoss):
            
            H = model.fit(
                trainAug.flow(trainX, trainY, batch_size=self.BS),
                steps_per_epoch=len(trainX) // self.BS,
                validation_data=(testX, testY),
                validation_steps=len(testX) // self.BS,
                class_weight=dict(enumerate(self.class_weights)),
                epochs=self.EPOCHS,
                callbacks=callback)
        else:
            H = model.fit(
                trainAug.flow(trainX, trainY, batch_size=self.BS),
                steps_per_epoch=len(trainX) // self.BS,
                validation_data=(testX, testY),
                validation_steps=len(testX) // self.BS,
                epochs=self.EPOCHS,
                callbacks=callback)
        # Stop Timer
        stop = timeit.default_timer()
        print('Total Training Time: ', stop - start) 
        # Adding to metadata
        self.meta["traingTime"] = stop - start
        self.meta["earlyStop"] = earlyStop
        self.meta["learningDecay"] = learningDecay
        self.meta["wloss"] = weightedLoss
        self.meta["epochs"] = len(H.history["loss"])
        return (H, model)

    def startTesting(self, testX, testY, model, voting=0):
        # Make predictions on the testing set
        # 테스트 세트에서 예측한다
        print("Making Predictions on the Test Set")
        # adding meta information
        self.meta["voting"] = voting

        if not voting:
            predIdxs = model.predict(testX, batch_size=self.BS)
            predIdxs = np.argmax(predIdxs, axis=1)
            return predIdxs
        else:
            # If voting is selected
            print("[INFO]: Test by Voting Initiated")
            votedY = []
            # Image Generator
            dataGenArgs = dict(rotation_range=2,
                    zoom_range=0.1,
                    fill_mode='constant')
            dataGen = ImageDataGenerator(**dataGenArgs)
            for i in range(len(testX)):
                # Augment a single image into VotingSize images
                iterGen = dataGen.flow(np.array([testX[i]]), np.array([testY[i]]), batch_size=1, seed=1)
                (genX, genY) = ([], [])

                # init Vote counter
                voteCounter = {}
                voteCounter[0] = 0
                voteCounter[1] = 0
                voteCounter[2] = 0
                # Create augmented data
                for i in range(voting):
                    X, Y = iterGen.next()
                    genX.append(X[0])
                    genY.append(Y[0])

                genX = np.array(genX)
                genY = np.array(genY)
                #plt.imshow(genX[6])

                # perform Predition on images
                preds = model.predict(genX, batch_size=self.BS)
                preds = np.argmax(preds, axis=1)
                # Votes Selection
                for vote in preds:
                    if vote==0:
                        voteCounter[0] += 1
                    elif vote==1:
                        voteCounter[1] += 1
                    elif vote==2:
                        voteCounter[2] += 1
    
                # getting Voting Results
                res = list(voteCounter.values())
                #print(res)
                voted = np.array(res).argmax()
                votedY.append(voted)
            votedY = np.array(votedY)
            return votedY

    def evalModel(self, predIdxs, testY, cReport=True):
        print("[INFO]: Model Evaluation")
        
        print("Classification Report")
        if cReport:
            print(classification_report(testY.argmax(axis=1), predIdxs, target_names=self.lb.classes_))

        # Compute Confusion Matrix and derrive raw, accuracy, sensitivity, specificity from it
        # 혼란 매트릭스
        cm= confusion_matrix(testY.argmax(axis=1), predIdxs)
        total = sum(sum(cm))
        if self.meta["classification"]=="binary":
            acc = (cm[0,0] + cm[1,1]) / total
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            
        else:
            acc = (cm[0,0] + cm[1,1] + cm[2,2]) / total
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[0,2])
            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1] + cm[1,2])
            specificity2 = cm[2, 2] / (cm[2, 0] + cm[2, 1] + cm[2,2])
            specificity = (specificity + specificity2) / 2

        # show the confusion matrix, accuracy, sensitivity, and specificity
        # 혼란 매트릭스 보기
        print("Confusion Matrix and its Derrivatives")
        print(cm)
        print("acc: {:.4f}".format(acc))
        print("sensitivity: {:.4f}".format(sensitivity))
        print("specificity: {:.4f}".format(specificity))
        # Adding to metadata
        self.meta["accuracy"] = int(acc*100)

    def eval2(self, y, y_hat):
        count = 0
        for i in range(len(y)):
            if y[i]==y_hat[i]:
                count += 1
        acc = int((count / len(y) )* 100)
        self.meta["accuracy"] = acc


    def generatePlot(self, H, iterInfo=1, arstats=False):
        # plot the training loss and accuracy
        # 플롯 그래프
        print("[INFO]: Plot Generation")
        N = self.meta["epochs"]
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        title = "AR-"+self.meta["model"]+"-"+self.meta["dataInfo"]+"-lr_"+str(self.meta["learningRate"])+"-dropout_"+str(self.meta["dropoutRate"])+"-acc_"+str(self.meta["accuracy"])
        #plt.title("Allergic Rhinitis-Xception-aligned-0.5d")
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        if arstats:
            cwd = os.getcwd()
            figName = cwd+"/Figures/ARStats.Plots/" + str(self.meta["imgNumber"]) + ".png"
        else:
            figName = "[iter-"+str(iterInfo)+"]plot-" + datetime.now().strftime('%H-%M-%S')
        plt.savefig(figName)
        self.meta["imgNumber"] += 1

    def getGradCams(self, type, model, testX = None, alpha=0.4):

        # Set the last convolution layer
        lastConvLayer = "block14_sepconv2_act"

        # For all images in the dataset
        if type=="all":
            for img in self.imagePaths:
                # get heatmmap
                heat = getHeatMap(image=img, model=model, lastConvLayer=lastConvLayer, imageType="path")
                #save gradcam
                saveGradCam(image = img, heatmap = heat, alpha=alpha, imageType="path")

        if type=="test":
            for i in range(0, len(testX)):
                testImage = testX[i]
                testImage = np.expand_dims(testImage, axis=0)
                # Rescale image to a range 0-255
                testImage = np.uint(255 * testImage)
                outString = "gradCam-"+str(i)+".jpg"
                heat = getHeatMap(image=testImage, model=model, lastConvLayer=lastConvLayer, imageType="image")
                saveGradCam(image=testImage[0], heatmap = heat, outPath=outString, alpha=alpha, imageType="image")

    def updateCSV(self):
        self.fileOP = open("ARStats.csv", "a")
        line = str(self.meta["imgNumber"]) + ","
        line += self.meta["imageType"] + "," + self.meta["model"] + "," + self.meta["loss"] + "," + self.meta["classType"] + ","
        line += str(self.meta["colorMode"]) +"," + str(self.meta["clean"]) + "," + str(self.meta["crop"]) + ","
        line += str(self.meta["resize"]) + "," + str(self.meta["colorCorrect"]) + "," + str(self.meta["baseTrain"]) + ","
        line += str(self.meta["activation"]) + "," + str(self.meta["wloss"]) + "," + str(self.meta["learningRate"]) + ","
        line += str(self.meta["learningDecay"]) + "," + str(self.meta["earlyStop"]) + "," + str(self.meta["epochsMax"]) + ","
        line += str(self.meta["epochs"]) + "," + str(self.meta["voting"]) + "," + str(self.meta["traingTime"]) + "," + str(self.meta["accuracy"])
        line += "\n"
        self.fileOP.write(line)
        self.fileOP.close()

    def crossValidate(self, path = r'C:\Users\cvpr\Documents\Bishal\Allergic Rhinitis\Dataset', crop=False, 
                    clean=False, modelType="inception", loss="bce", activation="relu",
                    dropoutRate=0.5, batchSize=8, epochs=100, learningRate=1e-3, iter=2, dataType="all", 
                    baseTrainable=False, weightedLoss=False, learningDecay=False, earlyStop=False, voting=0, arstats=False):
        '''
        Cross Validation by picking one dataset as test and rest as train at one time.
        '''
        dirList = {}
        imagePaths = list(paths.list_images(path))
        self.meta["dataInfo"] = dataType

        for imagePath in imagePaths:
            dataset = imagePath.split(os.path.sep)[-3]
            if dataset not in list(dirList.keys()):
                dirList[dataset] = []
            dirList[dataset].append(imagePath)

        dirListKeys = list(dirList.keys())
        
        for data_i in range(0, len(dirListKeys)):
            testSetData = []
            trainSetData = []
            for data_j in range(0, len(dirListKeys)):
                if data_i == data_j:
                    testSetData = testSetData + dirList[dirListKeys[data_j]]
                else:
                    trainSetData = trainSetData + dirList[dirListKeys[data_j]]
            
            print("Cross Validation %d - testSet="%data_i,dirListKeys[data_i])

            trainData = []
            trainLabels = []
            testData = []
            testLabels = []

            print("[INFO]: Trying to Read the images")
            for imagePath in trainSetData:
                label = imagePath.split(os.path.sep)[-2]
                trainLabels.append(label)
                image = cv2.imread(imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if clean:
                    image = cleanImage(image)
                if crop:
                    image = cropImage(image)
                    image = resizeImage(image, (224,224))
                # Append to data
                trainData.append(image)
            print("Train Images found :", len(trainData))

            for imagePath in testSetData:
                label = imagePath.split(os.path.sep)[-2]
                testLabels.append(label)
                image = cv2.imread(imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if clean:
                    image = cleanImage(image)
                if crop:
                    image = cropImage(image)
                    image = resizeImage(image, (224,224))
                # Append to data
                testData.append(image)
            print("Test Images found :", len(testData))

            
            print("[INFO]: Preparing Data")
            trainData = np.array(trainData) / 255.0
            trainLabels = np.array(trainLabels)

            self.meta["imgDim"] = trainData[0].shape
            self.meta["imageCount"] = len(trainData)
            self.meta["imageType"] = dataType
            self.meta["classType"] = "multiclass"
            self.meta["colorMode"] = "RGB"
            self.meta["clean"] = clean
            self.meta["crop"] = crop
            self.meta["resize"] = crop
            self.meta["colorCorrect"] = False
            self.meta["epochsMax"] = epochs

            # Computing Class Weights
            self.class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(testLabels), y=testLabels)
            self.meta["weightedLoss"] = self.class_weights

            self.lb = LabelBinarizer()
            trainLabels = self.lb.fit_transform(trainLabels)

            testData = np.array(testData) / 255.0
            testLabels = np.array(testLabels)

            testLabels = self.lb.fit_transform(testLabels)
            


            # To debug only data preparation part
            runModel = True

            if runModel:
                print("[INFO]: Augmenting Data - ")
                # Data Augmentation
                trainAug = self.setDataAugmentation(normalizeData=False, rotate=2, zoom=0.15, wShift=0.2, hShift=0.2, 
                                    shear=0.15, hFlip=True, vFlip=False, generateImages=True)   

                print("[INFO]: Loading Model")
                self.meta["classification"]="multiclass"
                # Set Base Model
                currBModel = self.setBaseModel(modelType)
                # Set Head Model
                currHModel = self.setHeadModel(currBModel, dropoutRate=dropoutRate, activation=activation)
                # Set final Model
                finalModel = self.initModel(currBModel, currHModel, baseTrainable=baseTrainable)

                print("[INFO]: Setting HyperParameters")
                self.INIT_LR = learningRate
                self.EPOCHS = epochs
                self.BS = batchSize

                self.meta["learningRate"] = learningRate
                self.meta["dropoutRate"] = dropoutRate
        
                print("[INFO]: Compiling Model")
                finalModel = self.compileModel(finalModel, loss=loss)

                print("[INFO] Model Training")
                # Start training
                (H, finalModel) = self.startTraining(finalModel, trainAug, trainData, trainLabels, testData, testLabels, 
                                                        weightedLoss=True, learningDecay= learningDecay, earlyStop=earlyStop)
               
               
               
               
                # Start Testing
                predIdxs = self.startTesting(testData, testLabels, finalModel, voting)
  

                print("[INFO]: Model Evaluation")
                # Evaluate Model based on test outputs
                self.evalModel(predIdxs, testLabels)

                # add information to accuracy file
                self.updateCSV()

                print("[INFO]: Plot Generation")
                if arstats:
                    self.generatePlot(H, iterInfo="CV", arstats=arstats)
                else:
                    self.generatePlot(H, iterInfo="CV-"+str(iter))




            