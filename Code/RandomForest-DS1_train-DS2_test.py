import numpy as np
import matplotlib.pyplot as plt
import csv
import random as rnd
from sklearn.ensemble import RandomForestRegressor


# Reads in raw data and returns it in a list of lists
def importData(filename): 
    data = []

    with open(filename) as myFile:
    #with open('Data/sanvito_ct.csv') as myFile:
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

    #print(data[0])
    return data

# Separates data into a training set and a test set
def selectData():
    train = importData('Data/DS1.csv')
    test = importData('Data/DS2.csv')


    print('Training set size = ', len(train))
    print('Test set size = ', len(test))

    return train[1:], test[1:]

def randForest():
    train, test = selectData()
    trainMat = []
    trainTc = []
    print('Imported Data')
    for i in train:
        trainTc.append(i[1])
        trainMat.append(i[2::])
    X = np.array(trainMat)
    y = np.array(trainTc)
    print('Made array')
    # define the model
    model = RandomForestRegressor(random_state=30)
    # fit the model on the whole dataset
    model.fit(X, y)
    
    from pprint import pprint
    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(model.get_params())

    chemComp = []
    testMat = []
    testTc = []
    for i in test:
        testTc.append(i[1])
        chemComp.append(i[0])
        testMat.append(i[2::])
    # make predictions
    yhat = model.predict(testMat)
    
    return yhat,testTc, chemComp

def calcPlotError():
    pred, real, composition = randForest()
    b50 = 0
    b100 = 0
    errors = 0
    count = 0 
    for i in range(len(pred)):
        trueTC = real[i]

        err = abs(pred[i] - trueTC)
        if err < 50:
            b50 = b50 + 1
        if err < 100:
            b100 = b100 + 1
        errors = errors + err
        count += 1
    MAE = (errors/len(pred))

    b50 = (b50/count)*100
    b100 = (b100/count)*100
    avg = sum(real)/len(real)
    num = 0
    den = 0
    for i in range(len(real)):
        num = (real[i] - pred[i])**2 + num
        den = (real[i] - avg)**2 + den

    r2 = 1 - (num/den)


    print('Random Forest finished with a ',MAE,' kelvin mean absolute error')
    print('Random Forest finished with a ',r2,' R^2 score')

    #plt.figure(figsize=(7.5, 7.5))
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

    '''for i in range(len(pred)):
        for j in range(len(pred)):
            if i != j:
                if pred[i] == pred[j]:
                    print('These two compounds have the same predictions: ' + composition[i] + ' and ' + composition[j])'''


def errorTCReal():
    pred, real, composition = randForest()
    
    errors = []
    b = []
    A = []

    for i in range(len(pred)):
        trueTC = real[i]

        err = trueTC - (pred[i])
        errors.append(err)
        A.append([1, real[i]])
        b.append([err])


    overx = []
    overy = []
    underx = []
    undery = []

    for i in range(len(errors)):
        if errors[i] >= 0:
            underx.append(real[i])
            undery.append(errors[i])
        else:
            overx.append(real[i])
            overy.append(errors[i])
        

    A = np.array(A)
    b = np.array(b)

    pinv = np.linalg.pinv(A)

    xmat = np.matmul(pinv, b)

    print("INTERCEPT AND SLOPE", xmat)

        

    # We can set the number of bins with the *bins* keyword argument.
    plt.figure(figsize=(9, 7))
    plt.rcParams.update({'font.size': 18})
    plt.axline((0, xmat[0][0]), slope= xmat[1][0], color= 'black')
    #plt.ylim(-400,800)
    plt.scatter(overx, overy, marker = '.', c = 'b', label = 'Model Overestimates')
    plt.scatter(underx, undery, marker = '.', c = 'r', label = 'Model Underestimates')
    plt.xlabel('Experimental $T_\mathrm{C}$ (K)')
    
    
    plt.legend(loc = 'upper left', markerscale = 3)
    plt.ylabel('Experimental - Predicted $T_\mathrm{C}$ Error (K)')
    plt.title('Random Forest Experimental $T_\mathrm{C}$ vs Error')
    


def errorTCPredicted():
    pred, real, composiiton = randForest()
    
    errors = []
    b = []
    A = []

    for i in range(len(pred)):
        trueTC = real[i]

        err = trueTC - (pred[i])
        errors.append(err)
        A.append([1, pred[i]])
        
        b.append([err])


    overx = []
    overy = []
    underx = []
    undery = []


    for i in range(len(errors)):
        if errors[i] >= 0:
            underx.append(pred[i])
            undery.append(errors[i])
        else:
            overx.append(pred[i])
            overy.append(errors[i])
        

    A = np.array(A)
    b = np.array(b)

    pinv = np.linalg.pinv(A)

    xmat = np.matmul(pinv, b)

    print("INTERCEPT AND SLOPE", xmat)


    # We can set the number of bins with the *bins* keyword argument.
    plt.figure(figsize=(9, 7))
    plt.rcParams.update({'font.size': 18})
    plt.axline((0, xmat[0][0]), slope= xmat[1][0], color= 'black')
    #plt.ylim(-400,800)
    plt.scatter(overx, overy, marker = '.', c = 'b', label = 'Model Overestimates')
    plt.scatter(underx, undery, marker = '.', c = 'r', label = 'Model Underestimates')

    plt.xlabel('Predicted $T_\mathrm{C}$ (K)')
    plt.legend(loc = 'upper right', markerscale = 3)

    plt.ylabel('Experimental - Predicted $T_\mathrm{C}$ Error (K)')
    plt.title('Random Forest Predicted $T_\mathrm{C}$ vs Error')


calcPlotError()
errorTCPredicted()
errorTCReal()

plt.show()