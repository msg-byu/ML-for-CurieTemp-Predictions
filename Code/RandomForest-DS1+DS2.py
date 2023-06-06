import numpy as np
import matplotlib.pyplot as plt
import csv
import statistics as sts
import random as rnd
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split


HEADER = ['Name', 'Temp', 'Ag', 'Al', 'Am', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 
    'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr', 'Cs', 'Cu', 'D', 'Dy', 'Er', 'Eu', 'F', 'Fe', 'Ga', 'Gd', 
    'Ge', 'H', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 
    'Na', 'Nb', 'Nd', 'Ni', 'Np', 'O', 'Os', 'P', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 
    'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'T', 'Ta', 'Tb', 'Te', 'Th', 'Ti', 
    'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

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

    '''hightc = [data[0]]
    tc = []

    for i in data[1:]:
        
        if i[1] > 600:
            hightc.append(i)
            tc.append(i[1])

    print('Average is: ', sts.mean(tc))'''
    return data

def createCompound():
    e1 = 'Co'
    e2 = 'Fe'
    e3 = 'Si'

    a1 = 0.4
    a2 = 0.25
    a3 = 0.35

    row = np.zeros(87).tolist()

    row[getHeaderIndex(e1) - 2] = a1
    row[getHeaderIndex(e2) - 2] = a2
    row[getHeaderIndex(e3) - 2] = a3

    return(pd.DataFrame(row).T)




def addExtraFeature(data):
    for i in range(len(data)):
        if i == 0:
            data[i].append('one')
        else:
            data[i].append(1)

    print(data[1])

    return data



# Separates data into a training set and a test set
def selectData():

    lst = importData("Data/DS1+DS2.csv")

    # making dataframe 
    df = pd.DataFrame(lst[1:], columns=lst[0])
    ndf = df.loc[:, df.columns!='Name']
    

    y = ndf['TC']
    X = ndf.loc[:, ndf.columns!='TC']


    X_train, X_test, y_train, y_test = train_test_split(X,y, 
                            random_state=104,
                            train_size=0.67)
    


    print('Training set size = ', len(X_train))
    print('Test set size = ', len(X_test))

    return X_train, X_test, y_train, y_test



def randForest():
    X_train, X_test, y_train, y_test = selectData()

    print('Made array')
    # define the model
    model = RandomForestRegressor(random_state= 30)
    # fit the model on the whole dataset
    model.fit(X_train, y_train)
    
    from pprint import pprint
    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(model.get_params())

    yhat = model.predict(X_test)
    print("prediction")
    print(yhat)
    
    return yhat,y_test

def calcPlotError():
    pred, real = randForest()

    real = real.tolist()

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
    plt.text(700, 1400, f'%d%% within 50 K' % b50, fontsize = 18)
    plt.text(700, 1300, f'%d%% within 100 K' % b100, fontsize = 18)





def errorTCPredicted():
    pred, real = randForest()

    real = real.tolist()
    
    errors = []
    b = []
    A = []

    for i in range(len(pred)):

        err = real[i] - pred[i]
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
    #plt.xlim(600,1400)
    plt.scatter(overx, overy, marker = '.', c = 'b', label = 'Model Overestimates')
    plt.scatter(underx, undery, marker = '.', c = 'r', label = 'Model Underestimates')
    plt.xlabel('Predicted $T_\mathrm{C}$ (K)')
    plt.legend(loc = 'upper right', markerscale = 3)
    plt.ylabel('Experimental - Predicted $T_\mathrm{C}$ Error (K)')
    plt.title('Random Forest Predicted $T_\mathrm{C}$ vs Error')


def errorTCReal():
    pred, real = randForest()

    real = real.tolist()
    
    errors = []
    b = []
    A = []

    for i in range(len(pred)):

        err = real[i] - pred[i]
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
    #plt.xlim(600,1400)
    plt.scatter(overx, overy, marker = '.', c = 'b', label = 'Model Overestimates')
    plt.scatter(underx, undery, marker = '.', c = 'r', label = 'Model Underestimates')
    plt.xlabel('Experimental $T_\mathrm{C}$ (K)')
    plt.legend(loc = 'upper left', markerscale = 3)
    plt.ylabel('Experimental - Predicted $T_\mathrm{C}$ Error (K)')
    plt.title('Random Forest Real $T_\mathrm{C}$ vs Error')



calcPlotError()
errorTCPredicted()
errorTCReal()

plt.show()