import numpy as np
import matplotlib.pyplot as plt
import csv
import statistics as sts
import random as rnd
from sklearn.decomposition import PCA
# random forest for making predictions for regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# Reads in raw data and returns it in a list of lists
def importFormattedData(): 
    data = []

    with open('Data/DS1.csv') as myFile:
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

    #print(data)
    return data

# Separates data into a training set and a test set
def selectData(data):
    rnd.seed(50) #97, 70, 50
    train = []
    sampleSize = 767

    test = rnd.sample(data, sampleSize)
    #control = data

    print('Sample size = ', len(test))
    print('Data size = ', len(data))

    for i in range(len(data)):
        boo = False
        for j in range(len(test)):
            if data[i][0] == test[j][0] and data[i][1] == test[j][1]:
                boo = True
        if boo == False:
            train.append(data[i])
    
    print('Training set size = ', len(train))

    print('Data Separated')

    return train, test


def pcaMethod(sz):
    matrix = importFormattedData()
    newMat = []
    tc = []
    print('Imported Data')
    for i in matrix[1::]:
        tc.append(i[1])
        newMat.append(i[2::])
    A = np.array(newMat)
    print('Made array')
    #print(A)
    # create the PCA instance
    pca = PCA(n_components=sz)
    # fit on data
    pca.fit(A)
    # transform data
    redMat = pca.transform(A)
    print('Ran PCA')

    mat = []
    redMat = redMat.tolist()

    for i in range(len(tc)):
        newl = [tc[i]]
        for j in redMat[i]:
            newl.append(j)
        mat.append(newl)

    print('Done with PCA')

    return mat


def randForest(sz):
    matrix = pcaMethod(sz)
    control, test = selectData(matrix)
    trainMat = []
    trainTc = []
    print('Imported PCA')
    for i in control:
        #print(i)
        trainTc.append(i[0])
        trainMat.append(i[1::])
    X = np.array(trainMat)
    y = np.array(trainTc)
    print('Made X and y array')
    # define the model
    model = RandomForestRegressor(random_state=30)
    # fit the model on the whole dataset
    model.fit(X, y)
    # make predictions
    from pprint import pprint
    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(model.get_params())

    testMat = []
    testTc = []
    for i in test:
        testTc.append(i[0])
        testMat.append(i[1::])

    yhat = model.predict(testMat)
    # summarize prediction
    return yhat,testTc

def calcPlotError(sz):
    pred, real = randForest(sz)
    b50 = 0
    b100 = 0
    errors = 0
    for i in range(len(pred)):
        trueTC = real[i]
        err = abs(pred[i] - trueTC)
        if err < 50:
            b50 = b50 + 1
        if err < 100:
            b100 = b100 + 1
        errors = errors + err
    MAE = (errors/len(pred))

    b50 = (b50/len(pred))*100
    b100 = (b100/len(pred))*100
    avg = sum(real)/len(real)
    num = 0
    den = 0
    for i in range(len(real)):
        num = (real[i] - pred[i])**2 + num
        den = (real[i] - avg)**2 + den

    r2 = 1 - (num/den)


    print('Random Forest finished with a ',MAE,' kelvin mean absolute error')
    print('Random Forest finished with a ',r2,' R^2 score')

    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 18})
    plt.scatter(pred,real,marker=".")
    plt.plot([0,1400],[0,1400],color='red')
    plt.xlabel('Predicted $T_\mathrm{C}$ (K)')
    plt.ylabel('Experimental $T_\mathrm{C}$ (K)')
    plt.title('Random Forest Prediction')
    #plt.text(100, 1400, f'%d Kelvin Mean Average Error' % MAE, fontsize = 12)
    plt.text(100, 1400, f'%d%% within 50 K' % b50, fontsize = 18)
    plt.text(100, 1300, f'%d%% within 100 K' % b100, fontsize = 18)
    

def findBestSize():
    x = range(5,85)

    errors = []
    for i in x:
        print('For Columns = ',i)
        pred, real = randForest(i)
        mis = 0
        for i in range(len(pred)):
            trueTC = real[i]
            err = abs(pred[i] - trueTC)
            mis = mis + err
        MAE = (mis/len(pred))
        errors.append(MAE)

    print(errors)
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(7.5, 8))
    plt.plot(x,errors, marker = 'o')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Number of Features')
    #plt.xticks(np.arange(5, 80, 1.0))
    plt.title('Number of Features vs Mean Absolute Error')
    plt.savefig("./Plots/PCA Random Forest - DS1.png", bbox_inches='tight')





findBestSize()
plt.show()