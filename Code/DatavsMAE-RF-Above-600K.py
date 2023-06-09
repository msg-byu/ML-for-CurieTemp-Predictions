import numpy as np
import matplotlib.pyplot as plt
import csv
import statistics as sts
import random as rnd
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split


HEADER = ['Name', 'Temp', 'Ag', 'Al', 'Am', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Bk', 'Br', 'C', 'Ca', 
            'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F',
              'Fe', 'Ga', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 'K', 'La',
                'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'Np', 'O', 'Os',
                  'P', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb',
                    'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'Ti', 'Tl',
                      'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']


def filterTC(data):
    newdata = []
    for i in data:
        if i[1] > 600:
            newdata.append(i)

    print(len(newdata), " compounds above 600K")

    return newdata


def getHeaderIndex(elem):
    for i in range(len(HEADER)):
        if elem == HEADER[i]:
            return i
    return "ERROR"

# Reads in raw data and returns it in a list of lists
def importData(filename): 
    data = []

    with open(filename) as myFile:
        csvdata = csv.reader(myFile, delimiter = ',')

        for i in csvdata:
            data.append(i)

    myFile.close()

    lend = len(data)
    lenr = len(data[0])

    for row in range(lend):
        if row == 0:
            continue
        for ele in range(lenr):
            if ele > 0:
                data[row][ele] = float(data[row][ele])
    return data


design = importData("Data/DS1+DS2.csv")
DATA = filterTC(design[1:])
DATA.insert(0,design[0])
TEST = 322
MAX_DATA = (len(DATA) - 1) - TEST
print(TEST)
print(MAX_DATA)




# Separates data into a training set and a test set
def selectData(ss, seed):

    lst = DATA.copy()


    rnd.seed(seed) #50
    train = []
    
    sampleSize = TEST

    test = rnd.sample(lst[1:], sampleSize)


    #print('Sample size = ', len(test))
    #print('Data size = ', len(lst))

    for i in range(1,len(lst)):
        boo = False
        for j in range(len(test)):
            if lst[i][0] == test[j][0] and lst[i][1] == test[j][1]:
                boo = True
        if boo == False:
            train.append(lst[i])

    #MAX_DATA = len(train)

    print('SS IS: ',ss)
    print('TRAIN SIZE IS: ', len(train))

    trainSample = rnd.sample(train, ss)

    
    #print('Training set size = ', len(trainSample))

    

    return trainSample, test

def randForest(ss, seed):
    train, test = selectData(ss, seed)

    trainMat = []
    trainTc = []
    #print('Imported Data')
    for i in train:
        trainTc.append(i[1])
        trainMat.append(i[2::])
    X = np.array(trainMat)
    y = np.array(trainTc)
    #print('Made array')
    # define the model
    model = RandomForestRegressor()
    # fit the model on the whole dataset
    model.fit(X, y)
    # make predictions
    from pprint import pprint

    testMat = []
    testTc = []
    for i in test:
        testTc.append(i[1])
        testMat.append(i[2::])

    yhat = model.predict(testMat)
    # summarize prediction


    return yhat,testTc


def calcMAE(yhat,y):
    errors = 0

    for i in range(len(yhat)):
        err = abs(yhat[i] - y[i])
        errors = errors + err
    MAE = (errors/len(yhat))

    #print('Random Forest finished with a ',MAE,' kelvin mean absolute error')

    return MAE


def trainVSMAE():
    trainingSize = list(range(322, 652, 10))
    MAESmat =[]

    for j in range(10):
        MAES = []
        seed = j*10

        for i in range(322, 652, 10):
            ss = i
            if i == 642:
                ss = MAX_DATA
            
            yhat, y = randForest(ss,seed)
            MAE = calcMAE(yhat, y)

            MAES.append(MAE)

        MAESmat.append(MAES)


    for i in MAESmat:
        plt.plot(trainingSize, i)
    plt.xlabel('Training Data Size')
    plt.ylabel('MAE')
    plt.title('Training Data Size vs MAE')

    
    plt.show()



def trainVSMAEavg():
    trainingSize = list(range(322, 652, 10))
    MAESmat =[]

    for j in range(100):
        print("ROUND: ", j)
        MAES = []
        seed = j*10

        for i in range(322, 652, 10):
            ss = i
            if i > 642:
                ss = MAX_DATA
            
            yhat, y = randForest(ss,seed)
            MAE = calcMAE(yhat, y)

            MAES.append(MAE)

        MAESmat.append(MAES)

    Y = []
    for i in range(len(trainingSize)):
        temp = []
        for j in range(len(MAESmat)):
            temp.append(MAESmat[j][i])

        avg = sts.mean(temp)
        Y.append(avg)
    
    print(trainingSize)
    print(Y)
    #plt.figure(figsize=(7.5, 7.5))
    plt.rcParams.update({'font.size': 12})
    plt.plot(trainingSize, Y)
    plt.xlabel('Training Data Size')
    plt.ylabel('MAE')
    plt.title('Training Data Size vs MAE Above 600K')

    plt.savefig('./Plots/MAE vs Training Data Size Above 600K.png', bbox_inches='tight')


def plotPlot():
    
    x = [322, 332, 342, 352, 362, 372, 382, 392, 402, 412, 422, 432, 442, 452, 462, 472, 482, 492, 502, 512, 522, 532, 542, 552, 562, 572, 582, 592, 602, 612, 622, 632, 642]
    y = [69.93116051242237, 69.60070057453417, 68.9750806832298, 68.79397116459629, 68.56912051242236, 67.99862649068324, 67.59270948757764, 67.26946821428571, 
         66.8736645652174, 66.44236189440994, 66.22640701863354, 65.80585905279504, 65.57348431677019, 65.18397276397515, 64.84540187888199, 64.57136864906832, 
         64.23899357142857, 63.82146402173913, 63.57965350931677, 63.21466675465839, 63.002193804347826, 62.57664309006212, 62.35237681677019, 62.10436394409938, 
         61.739175263975156, 61.59302172360249, 61.40661597826087, 61.174254596273286, 60.84736610248447, 60.618130838509316, 60.51390923913044, 60.1557101552795,
           60.046010714285714]
    
    
    plt.rcParams.update({'font.size': 12})
    plt.plot(x, y)
    plt.scatter(x,y)
    plt.xlabel('Training Data Size')
    plt.ylabel('MAE (K)')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Training Data Size vs MAE Above 600K')

    plt.savefig('./Plots/MAE vs Training Data Above 600K loglog.png', bbox_inches='tight')



def LogLog(x, S, I):
    y = I*(x**S)
    return y



trainVSMAEavg()
plotPlot()

plt.show()