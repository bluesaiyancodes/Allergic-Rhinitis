import cv2
import pandas as pd
import numpy as np

dataF = pd.read_csv('ARStats.csv')

data = dataF.to_numpy()

def calAcc(workData):
    # returns accuracy of Inception and Xception models when given inputs for the same
    vitals = {}
    vitals["accInc"] = []
    vitals["accXc"] = []
    vitals["timeIn"] = []
    vitals["timeX"] = []
    for i in range(len(workData)):
        if workData[i][2] == 'inception':
            vitals["accInc"].append(workData[i][-1])
            vitals["timeIn"].append(workData[i][-2])
        elif workData[i][2] == "xception":
            vitals["accXc"].append(workData[i][-1])
            vitals["timeX"].append(workData[i][-2])
    vitals["accIncMean"] = np.mean(vitals.get("accInc"))
    vitals["accIncStd"] = np.std(vitals.get("accInc"))
    vitals["accXcMean"] = np.mean(vitals.get("accXc"))
    vitals["accXcStd"] = np.std(vitals.get("accXc"))
    vitals["time"] = np.sum(vitals.get("timeIn")) / 60
    return vitals

def getOut(vitals):
    print("Inception - %.2f%% (+/- %.2f%%) time - %d" % (vitals.get("accIncMean"), vitals.get("accIncStd"), vitals.get("time")))  
    print("Xception  - %.2f%% (+/- %.2f%%) time - %d" % (vitals.get("accXcMean"), vitals.get("accXcStd"), vitals.get("time")))  

def setMetaData():
    metaData = {}
    metaData["No Voting"] = {}
    metaData.get("No Voting")["Non-Aligned"] = [134, 154]
    metaData.get("No Voting")["Aligned"] = [113, 134]
    metaData["10 Voting"] = {}
    metaData.get("10 Voting")["Non-Aligned"] = [234, 254]
    metaData.get("10 Voting")["Aligned"] = [254, 274]
    metaData["20 Voting"] = {}
    metaData.get("20 Voting")["Non-Aligned"] = [194, 214]
    metaData.get("20 Voting")["Aligned"] = [214, 234]
    metaData["30 Voting"] = {}
    metaData.get("30 Voting")["Non-Aligned"] = [154, 174]
    metaData.get("30 Voting")["Aligned"] = [174, 194]


    return metaData



metaData = setMetaData()
for opType in metaData.keys():
    print("")
    print(opType)
    for imType in metaData.get(opType).keys():
        print(imType)
        workData = data[metaData.get(opType).get(imType)[0]: metaData.get(opType).get(imType)[1]]
        vitals = calAcc(workData)
        getOut(vitals)
