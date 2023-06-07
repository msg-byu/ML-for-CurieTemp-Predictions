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



DATA = importData("Data/DS1+DS2.csv")
TEST = 850
MAX_DATA = (len(DATA) - 2) - TEST
print(TEST)
print(MAX_DATA)




# Separates data into a training set and a test set
def selectData(ss, seed):

    lst = DATA.copy()


    rnd.seed(seed) #50
    train = []
    
    sampleSize = TEST

    test = rnd.sample(lst[1:], sampleSize)


    print('Sample size = ', len(test))
    print('Data size = ', len(lst))

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
    trainingSize = list(range(850, 3800, 100))
    MAESmat =[]

    for j in range(10):
        MAES = []
        seed = j*10

        for i in range(850, 3800, 100):
            ss = i
            if i == 3700:
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
    trainingSize = list(range(850, 3800, 100))
    MAESmat =[]

    for j in range(100):
        print("ROUND: ", j)
        MAES = []
        seed = j*10

        for i in range(850, 3800, 100):
            ss = i
            if i > 3700:
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
    plt.title('Training Data Size vs MAE')

    plt.savefig('./Plots/MAE vs Training Data Size 100.png', bbox_inches='tight')


def plotPlot():
    
    x = [850, 950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050, 2150, 2250, 2350, 2450, 2550, 2650, 2750, 2850, 2950, 3050, 3150, 3250, 3350, 3450, 3550, 3650, 3709]
    y = [95.56648551173137, 93.31863551442746, 91.27398513436619, 89.52779732409893, 88.01309949028915, 86.62583029767542, 85.20222245550107, 83.97788156846073, 82.80218484874969, 81.71393835389276, 
         80.66935641486074, 79.49632872307016, 78.63279164752011, 77.6996407146806, 76.75638963961408, 75.87077599782266, 75.16320479026548, 74.56232739818394, 73.82554221829628, 73.09416364986099, 
         72.49726301668952, 72.00902978913389, 71.33166659711854, 70.75078997180123, 70.25819546489105, 69.83565674545912, 69.33783440021337, 68.95214083301191, 68.43487146504329, 68.24587728415285]
    
    slope = -0.224
    h = 431.4

    x2 = np.linspace(850, 3750, 100)
    x2[-1] = 3709
    y2 = h * (x2**slope)
    print(y2)

    
    plt.rcParams.update({'font.size': 12})
    plt.plot(x, y)
    plt.scatter(x,y)
    #plt.plot(x2, y2)
    plt.xlabel('Training Data Size')
    plt.ylabel('MAE')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Training Data Size vs MAE')

    plt.savefig('./Plots/MAE vs Training Data Size 100 loglog.png', bbox_inches='tight')



def LogLog(x, S, I):
    y = I*(x**S)
    return y

def fitData():

    x = [850, 950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050, 2150, 2250, 2350, 2450, 2550, 2650, 2750, 2850, 2950, 3050, 3150, 3250, 3350, 3450, 3550, 3650, 3710]
    y = [95.56648551173137, 93.31863551442746, 91.27398513436619, 89.52779732409893, 88.01309949028915, 86.62583029767542, 85.20222245550107, 83.97788156846073, 82.80218484874969, 81.71393835389276, 
         80.66935641486074, 79.49632872307016, 78.63279164752011, 77.6996407146806, 76.75638963961408, 75.87077599782266, 75.16320479026548, 74.56232739818394, 73.82554221829628, 73.09416364986099, 
         72.49726301668952, 72.00902978913389, 71.33166659711854, 70.75078997180123, 70.25819546489105, 69.83565674545912, 69.33783440021337, 68.95214083301191, 68.43487146504329, 68.24587728415285]

    import scipy.optimize as opt
    parameters, covariance = opt.curve_fit(LogLog, x, y)
    fit_S = parameters[0]
    fit_I = parameters[1]
    print('Slope: ', fit_S)
    print('Intercept: ',fit_I)

    fit_y = LogLog(x, fit_S, fit_I)

    x = np.log(x)
    y = np.log(y)
    fit_y = np.log(fit_y)

    plt.plot(x, y, 'o', label='data')
    plt.plot(x, fit_y, '-', label='fit')

    plt.xlabel('Training Data Size')
    plt.ylabel('MAE')

    #plt.yscale('log')
    #plt.xscale('log')

    plt.legend()
    plt.show()


def trainVSMAE():
    trainingSize = list(range(850, 3800, 100))
    MAESmat =[]

    for j in range(10):
        MAES = []
        seed = j*10

        for i in range(850, 3800, 100):
            ss = i
            if i == 3700:
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


trainVSMAEavg()
plotPlot()

plt.show()