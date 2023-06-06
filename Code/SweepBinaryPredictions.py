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


matrix = importFormattedData()

# Separates data into a training set and a test set
def selectData(data):
    rnd.seed(77)
    control = []
    sampleSize = round(len(data)/3)

    temp = rnd.sample(data, sampleSize)
    #control = data

    print('Sample size = ', len(temp))
    print('Data size = ', len(data))

    for i in range(len(data)):
        boo = False
        for j in range(len(temp)):
            if data[i][0] == temp[j][0] and data[i][1] == temp[j][1]:
                boo = True
        if boo == False:
            control.append(data[i])
    
    print('Training set size = ', len(control))

    return control, temp

# Calculates the distance between two vectors
def calcDistance(a1, a2):
    m1 = a1[2::]
    m2 = a2
    summ = 0

    for i in range(len(m1)):
        dif = m2[i]-m1[i]
        sqr = dif**2
        summ = summ + sqr

    dist = np.sqrt(summ)

    return dist

# Finds the k nearest neighbors in the training set to each test vector and takes the average of their curie temperature
def fastKNNtrain(train, test, num):
    control = train
    samp = test
    
    count = 0
    averages = []

    for i in samp:
        distances = []
        n = 0
        for j in control:
            distancia = calcDistance(j,i)
            distances.append([j[1],distancia,j[0]])
            n += 1
        distances.sort(key = lambda x: x[1])

        knn = distances[1:num + 1] #Does not include point being predicted on

        avg = 0
        for k in knn:
            avg = avg + k[0]
        avg = avg/num
        averages.append(avg)
        count += 1

        #print(count, ' iterations')
    real =[]
    for i in range(len(averages)):
        trueTC = samp[i][1]
        real.append(trueTC)


    return averages, real


def randForest(control, test):
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
    model = RandomForestRegressor(max_depth=90, n_estimators=1800, min_samples_leaf = 1, min_samples_split = 2, random_state=30)
    # fit the model on the whole dataset
    model.fit(X, y)
    # make predictions
    from pprint import pprint
    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(model.get_params())

    testMat = []
    for i in test:
        testMat.append(i)

    yhat = model.predict(testMat)
    # summarize prediction
    return yhat


def createBinaryData(elem1, elem2):
    compound = [elem1, elem2]
    compInds = []

    for i in compound:
        for j in range(len(matrix[0])):
            if i == matrix[0][j]:
                compInds.append(j - 2)
                break

    binData = []
    
    for i in range(101):
        comp = np.zeros(87).tolist()
        num1 = i * 0.01
        num2 = (100 - i)*0.01
        comp[compInds[0]] = num1
        comp[compInds[1]] = num2

        binData.append(comp)

    return np.array(binData)


def compoundChecker(elem1, elem2):
    #Enter desired compound components
    compound = [elem1, elem2]
    compInds = []

    for i in compound:
        for j in range(len(matrix[0])):
            if i == matrix[0][j]:
                compInds.append(j)
                break

    print(compInds);

    similars = []

    for i in matrix[1::]:
        isPart = True
        compSize = 0

        if i[compInds[0]] == 1 or i[compInds[1]] == 1:
            similars.append(i)


        for j in i[2::]:
            if j > 0:
                compSize += 1

        if len(compound) == compSize:
            for j in compInds:
                if i[j] == 0:
                    isPart = False
            
            if isPart == True:
                similars.append(i)
    

    x = []
    y = []

    if len(similars) > 0:
        for i in similars:
            print("Compound name: ", i[0])
            print("Compound TC: ", i[1])
            y.append(i[1])
            x.append(round(i[compInds[0]],2) * 100)
            for j in range(len(compInds)):
                num = round(i[compInds[j]],2) * 100
                print("Compound is ", num, "percent ", compound[j] )
            print('\n')
    else:
        print("No similar compounds found in data")

    return x,y


def createOptimizePlot(elem1, elem2):
    control = importFormattedData()
    test = createBinaryData(elem1, elem2)
    
    pred = randForest(control[1::], test)
    xReal, yReal = compoundChecker(elem1, elem2)

    #plt.figure(figsize=(7.5, 7.5))
    plt.rcParams.update({'font.size': 18})
    plt.plot(range(101), pred)
    if len(xReal) > 0:
        plt.scatter(xReal, yReal, color = 'red')
    plt.xlabel('Percentage of ' + elem1)
    plt.ylabel('Curie Temp (K)')
    plt.title('Combination of ' + elem1 + ' and ' + elem2)

    plt.show()


def createRFSubPlot(elem1, elem2):
    control = importFormattedData()
    test = createBinaryData(elem1, elem2)
    pred = randForest(control[1::], test)
    xReal, yReal = compoundChecker(elem1, elem2)

    return pred, xReal, yReal

def createKNNSubPlot(elem1, elem2):
    control = importFormattedData()
    test = createBinaryData(elem1, elem2)
    pred, real = fastKNNtrain(control[1::], test, 2)
    xReal, yReal = compoundChecker(elem1, elem2)
 
    return pred, xReal, yReal


def makeSubplots():
    x = 4
    y = 2

    compounds = [[['Co','Fe'],['Fe','Ni']],
                [['Fe','O'],['Mn','Ni']],
                [['Ni','Sm'],['Ni','Tb']],
                [['Se','U'],['Ga','U']]]


    fig, axs = plt.subplots(x, y, figsize=(9,14))

    for i in range(x):
        for j in range(y):
            
            elem1 = compounds[i][j][0]
            elem2 = compounds[i][j][1]

            #Choose between KNN or RF
            pred, xReal, yReal = createRFSubPlot(elem1, elem2)
            axs[i, j].plot(range(101), pred, label = 'Predicted',color = 'black')
            if len(xReal) > 0:
                axs[i, j].scatter(xReal, yReal, color = 'red', label = 'Experimental')
            axs[i, j].set_xlabel('Percentage of ' + elem1, fontsize=14)#14
            axs[i, j].set_ylabel('Curie Temp (K)', fontsize=14)#14
            axs[i, j].set_title('Combination of ' + elem1 + ' and ' + elem2, fontsize=16)#16

    axs[0,0].set_facecolor('#FADBD8')
    axs[0,1].set_facecolor('#FADBD8')
    axs[0,0].set_ylim(0, 1450)
    axs[0,1].set_ylim(0, 1450)

    axs[1,0].set_facecolor('#FDEDEC')
    axs[1,1].set_facecolor('#FDEDEC')
    axs[1,0].set_ylim(0, 1100)
    axs[1,1].set_ylim(0, 1100)

    axs[2,0].set_facecolor('#EBF5FB')
    axs[2,1].set_facecolor('#EBF5FB')
    axs[2,0].set_ylim(0, 700)
    axs[2,1].set_ylim(0, 700)

    axs[3,0].set_facecolor('#D6EAF8')
    axs[3,1].set_facecolor('#D6EAF8')
    axs[3,0].set_ylim(0, 200)
    axs[3,1].set_ylim(0, 200)

    axs[0, 0].legend(fontsize=11.5, loc = 'lower left')
    fig.tight_layout()
    #fig.savefig('./Curie Temp Plots/Final Figs/TCSweep.png', bbox_inches='tight')
    #fig.suptitle("KNN Sweep")
    #fig.savefig("KNN Prediction Sweep.png")
    plt.show()


def makeSubplotsCompare():
    x = 1
    y = 3

    compounds = [[['Fe','Ni'],['Co','Mn'],['Ni','Rh']]]

    fig, axs = plt.subplots(x, y, figsize=(4,12))

    for i in range(x):
        for j in range(y):
            
            elem1 = compounds[i][j][0]
            elem2 = compounds[i][j][1]

            pred, xReal, yReal = createSubPlot(elem1, elem2)
            axs[i, j].plot(range(101), pred, label = 'Predicted')
            if len(xReal) > 0:
                axs[i, j].scatter(xReal, yReal, color = 'red', label = 'Experimental')
            axs[i, j].set_xlabel('Percentage of ' + elem1, fontsize=14)
            axs[i, j].set_ylabel('Curie Temp (K)', fontsize=14)
            axs[i, j].set_title('Combination of ' + elem1 + ' and ' + elem2, fontsize=16)

    axs[0, 0].legend(fontsize=11.5, loc = 'upper left')
    fig.tight_layout()
    plt.show()

makeSubplots()


#makeSubplotsCompare()