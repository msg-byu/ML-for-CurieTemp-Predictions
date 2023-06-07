import numpy as np
import matplotlib.pyplot as plt
import csv
import random as rnd
# random forest for making predictions for regression
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

def binData(data):
    print("Dataset size is: ", len(data))
    rnd.seed(50)
    ss = 100;

    binned = []

    bin150 = []
    bin300 = []
    bin450 = []
    bin600 = []
    bin750 = []
    bin900 = []
    bin1050 = []
    bin1200 = []
    bin1350 = []
    rest = []

    for i in data:
        if i[1] < 150:
            print(i[1])
            bin150.append(i)
        elif i[1] < 300:
            bin300.append(i)
        elif i[1] < 450:
            bin450.append(i)
        elif i[1] < 600:
            bin600.append(i)
        elif i[1] < 750:
            bin750.append(i)
        elif i[1] < 900:
            bin900.append(i)
        elif i[1] < 1050:
            bin1050.append(i)
        elif i[1] < 1200:
            bin1200.append(i)
        elif i[1] < 1350:
            bin1350.append(i)
        else:
            rest.append(i)

    if len(bin150) > ss:
        binned.extend(rnd.sample(bin150, ss))
    else:
        binned.extend(bin150)

    if len(bin300) > ss:
        binned.extend(rnd.sample(bin300, ss))
    else:
        binned.extend(bin300)

    if len(bin450) > ss:
        binned.extend(rnd.sample(bin450, ss))
    else:
        binned.extend(bin450)

    if len(bin600) > ss:
        binned.extend(rnd.sample(bin600, ss))
    else:
        binned.extend(bin600)

    if len(bin750) > ss:
        binned.extend(rnd.sample(bin750, ss))
    else:
        binned.extend(bin750)

    if len(bin900) > ss:
        binned.extend(rnd.sample(bin900, ss))
    else:
        binned.extend(bin900)

    if len(bin1050) > ss:
        binned.extend(rnd.sample(bin1050, ss))
    else:
        binned.extend(bin1050)

    if len(bin1200) > ss:
        binned.extend(rnd.sample(bin1200, ss))
    else:
        binned.extend(bin1200)

    if len(bin1350) > ss:
        binned.extend(rnd.sample(bin1350, ss))
    else:
        binned.extend(bin1350)

    if len(rest) > ss:
        binned.extend(rnd.sample(rest, ss))
    else:
        binned.extend(rest)

    print("Binned size is: ", len(binned))

    return binned




# Separates data into a training set and a test set
def selectData(data):
    rnd.seed(50) #97, 70, 50
    train = []
    print("Dataset size is: ", len(data))
    sampleSize = round(len(data)/3)

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

    return train, test

def randForest():
    matrix = binData(importFormattedData()[1::])
    control, test = selectData(matrix)
    trainMat = []
    trainTc = []
    print('Imported Data')
    for i in control:
        trainTc.append(i[1])
        trainMat.append(i[2::])
    X = np.array(trainMat)
    y = np.array(trainTc)
    print('Made array')
    # define the model
    model = RandomForestRegressor(max_depth=90, n_estimators=1800, min_samples_leaf = 1, min_samples_split = 2, random_state= 30)
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
        testTc.append(i[1])
        testMat.append(i[2::])

    yhat = model.predict(testMat)
    # summarize prediction
    return yhat,testTc

def calcPlotError():
    pred, real = randForest()
    b50 = 0
    b100 = 0
    errors = 0
    count = 0
    for i in range(len(pred)):
        trueTC = real[i]
        if trueTC > 600:
            err = abs(pred[i] - trueTC)
            if err < 50:
                b50 = b50 + 1
            if err < 100:
                b100 = b100 + 1
            errors = errors + err
            count += 1
    MAE = (errors/count)

    b50 = (b50/count)*100
    b100 = (b100/count)*100
    avg = sum(real)/len(real)
    num = 0
    den = 0
    for i in range(len(real)):
        num = (real[i] - pred[i])**2 + num
        den = (real[i] - avg)**2 + den

    r2 = 1 - (num/den)


    print('Random Forest finished with a ',MAE,' kelvin mean absolute error Above 600K')
    print('Random Forest finished with a ',r2,' R^2 score')

    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 18})
    plt.scatter(pred,real,marker=".")
    plt.plot([0,1400],[0,1400],color='red')
    plt.xlabel('Predicted $T_\mathrm{C}$ (K)')
    plt.ylabel('Experimental $T_\mathrm{C}$ (K)')
    plt.title('Sub-dataset Random Forest Prediction')
    #plt.text(100, 1400, f'%d Kelvin Mean Average Error' % MAE, fontsize = 12)
    plt.text(0, 1400, f'%d%% within 50 K (above 600 K)' % b50, fontsize = 17)
    plt.text(0, 1300, f'%d%% within 100 K (above 600 K)' % b100, fontsize = 17)

    plt.savefig("./Plots/Even Distribution Random Forest DS1.png", bbox_inches='tight')





def errorTCReal():
    pred, real = randForest()
    
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
    plt.savefig("./Plots/Even Distribution Random Forest Experimental Error DS1.png", bbox_inches='tight')
    


def errorTCPredicted():
    pred, real = randForest()
    
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
    plt.savefig("./Plots/Even Distribution Random Forest Predicted Error DS1.png", bbox_inches='tight')


    

calcPlotError()
errorTCPredicted()
errorTCReal()

plt.show()





be = {'bootstrap': True, 'max_depth': 90, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1800}