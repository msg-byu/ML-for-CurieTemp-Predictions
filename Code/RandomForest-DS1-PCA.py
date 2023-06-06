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
    x = range(5,80)

    errors = []
    error = [85.64516144285086, 83.2583659278575, 82.21172003911343, 82.33426478487608, 79.8460977183833, 
    79.8637528031291, 79.42141275097785, 79.46714859191651, 80.36092273793996, 78.88444108213812, 
    79.58025293350721, 78.99592392438079, 81.1783298435463, 79.47306611473275, 79.69646560625816, 
    79.42085057366369, 79.81882473272482, 79.63460795306385, 79.95622144719687, 79.03510106910043, 
    78.64946796610162, 78.09864152542373, 79.25095949152545, 79.17057967405472, 78.60146204693612, 
    79.05559344198176, 77.91850324641463, 79.24579352020861, 78.26093104302478, 78.09354891786181, 
    77.95473119947857, 78.19563445893091, 78.26954148631027, 78.4954931942633, 77.86959560625817, 
    76.82879388526719, 76.57378632333779, 77.85449031290749, 77.15467658409398, 75.80327490221647, 
    77.85601638852671, 76.78409963494136, 75.77963658409384, 76.45591706649286, 75.97644035202086, 
    75.2736469361148, 75.48107185136901, 74.85985220338983, 75.28346155149933, 74.43791292046943, 
    74.93451255541069, 76.00976744458927, 76.06536362451111, 74.35542324641463, 75.66026916558016, 
    74.54080181225554, 74.36467292046935, 74.50127375488914, 74.30353212516299, 74.36458588005219, 
    74.88890020860504, 74.32785221642771, 73.77317290743147, 75.17892681442858, 74.4296975228161, 
    75.46853869621901, 75.77309187744454, 74.7602417514125, 75.39361070404168, 74.5096727553237, 
    74.95350906127764, 75.19141029986962, 75.10127677966094, 74.54892187744468, 75.2299554237288]
    '''for i in x:
        print('For Columns = ',i)
        pred, real = randForest(i)
        mis = 0
        for i in range(len(pred)):
            trueTC = real[i]
            err = abs(pred[i] - trueTC)
            mis = mis + err
        MAE = (mis/len(pred))
        errors.append(MAE)'''

    print(errors)
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(7.5, 8))
    plt.plot(x,error, marker = 'o')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Number of Features')
    #plt.xticks(np.arange(5, 80, 1.0))
    plt.title('Number of Features vs Mean Absolute Error')





findBestSize()
plt.show()