import numpy as np
import matplotlib.pyplot as plt
import csv
import statistics as sts
import random as rnd


# Reads in data and returns it in a list of lists
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
    rnd.seed(77)
    control = []
    #sampleSize = 767
    sampleSize = round(len(data)/3)

    temp = rnd.sample(data, sampleSize)

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
    print('Test set size = ', sampleSize)

    return control, temp

# Calculates the distance between two vectors
def calcDistance(a1, a2):
    m1 = a1[2::]
    m2 = a2[2::]
    summ = 0

    for i in range(len(m1)):
        dif = m2[i]-m1[i]
        sqr = dif**2
        summ = summ + sqr

    dist = np.sqrt(summ)

    return dist

# Finds the k nearest neighbors in the training set to each test vector and takes the average of their curie temperature
def fastKNNtrain(mat, num):
    [control, samp] = selectData(mat[1::])
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

    return averages,samp

# Counts the non magnetic compounds in the dataset
def countNonMag(matrix):
    count = 0
    for i in matrix:
        if i[1] == 0:
            count += 1
    
    print('Number of non magnetic materials: ', count)

# Predicts the curie temperatures of the test set and finds the mean relative area
def KNNpredict(num):

    newMat = importFormattedData()

    knn = num
    b50 = 0
    b100 = 0
    [avgs,samp] = fastKNNtrain(newMat, knn)


    errors = 0
    count = 0
    for i in range(len(avgs)):
        trueTC = samp[i][1]
        #if trueTC > 600:
        err = abs(avgs[i] - trueTC)
        if err < 50:
            b50 = b50 + 1
        if err < 100:
            b100 = b100 + 1
        errors = errors + err
        count +=1
    #MAE = (errors/len(avgs))
    MAE = (errors/count)

    b50 = (b50/count)*100
    b100 = (b100/count)*100

    print('KNN finished with a ',MAE,' kelvin mean absolute error')

    temps = []
    for i in samp:
        temps.append(i[1])

    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 18})
    plt.scatter(avgs,temps,marker=".")
    plt.plot([0,1400],[0,1400],color='red')
    plt.xlabel('Predicted $T_\mathrm{C}$ (K)')
    plt.ylabel('Experimental $T_\mathrm{C}$ (K)')
    plt.title(f'%d Nearest Neighbors Prediction ' % knn)
    #plt.text(100, 1400, f'%d Kelvin Mean Average Error' % MAE, fontsize = 12)
    plt.text(100, 1400, f'%d%% within 50 K' % b50, fontsize = 18)
    plt.text(100, 1300, f'%d%% within 100 K' % b100, fontsize = 18)

    return MAE

# Plots the mean absolute error for k values 1 through 20
def plotErrorRange():
    x = range(1,21)
    '''errors = [119.456673510467, 110.15719987922714, 108.96512855609242, 109.71060235507244, 110.19937866344597, 
    111.84665492485215, 114.02160720036801, 115.83154710144919, 117.04660945607446, 119.58868868760024, 
    121.68671402430118, 124.66893991545888, 126.44959606094409, 127.66134986772455, 
    129.28950158346808, 130.5426621628421, 132.1360987496447, 133.62770781893005, 135.3549417323504, 136.68576896135272]'''

    errors = []
    for i in x:
        print('For K = ',i)
        mae = KNNpredict(i)
        errors.append(mae)

    plt.plot(x,errors, marker = 'o')
    plt.ylabel('Mean Average Error')
    plt.xlabel('Number of Neighbors')
    plt.xticks(np.arange(1, 21, 1.0))
    plt.title('Number of Neighbors vs Mean Average Error')
    plt.show()

def randKNNtrain(mat, num):
    [control, samp] = selectData(mat[1::])
    count = 0
    averages = []
    rtemps = []

    rnd.seed(73)

    for i in control:
        rtemps.append(i[1])

    rnd.shuffle(rtemps)

    for i in samp:
        distances = []
        n = 0
        for j in control:
            distancia = calcDistance(j,i)
            distances.append([rtemps[n],distancia])
            n += 1
        distances.sort(key = lambda x: x[1])

        knn = distances[1:num + 1] #Does not include point being predicted on

        avg = 0
        for k in knn:
            avg = avg + k[0]
        avg = avg/num
        averages.append(avg)
        count += 1

        print(count, ' iterations')

    return averages,samp

def randKNNpredict(num):
    data = importFormattedData()

    knn = num
    [avgs,samp] = randKNNtrain(data, knn)
    errors = 0
    for i in range(len(avgs)):
        trueTC = samp[i][1]
        err = abs(avgs[i] - trueTC)
        errors = errors + err
    MAE = (errors/len(avgs))

    print('Random KNN finished with a ',MAE,' kelvin mean absolute error')
    temps = []
    for i in samp:
        temps.append(i[1])

    plt.scatter(avgs,temps)
    plt.plot([0,1400],[0,1400],color='red')
    plt.xlabel('Predicted Temp')
    plt.ylabel('Random Experimental Temp')
    plt.title(f'%d Nearest Neighbors Random Prediction' % knn)


    return MAE


def errorKNNPredicted():
    newMat = importFormattedData()
    knn = 2
    pred, real = fastKNNtrain(newMat, knn)

    temps = []

    errors = []
    b = []
    A = []
    for i in real:
        temps.append(i[1])


    for i in range(len(pred)):
        trueTC = temps[i]

        err = trueTC - (pred[i])
        errors.append(err)
        A.append([1, pred[i]])
        b.append([err])



    overx = []
    overy = []
    underx = []
    undery = []

    #Use predicted
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
    plt.scatter(overx, overy, marker = '.', c = 'b', label = 'Model Overestimates')
    plt.scatter(underx, undery, marker = '.', c = 'r', label = 'Model Underestimates')
    plt.xlabel('Predicted $T_\mathrm{C}$ (K)')
    plt.legend(loc = 'upper right', markerscale = 3)
    plt.ylabel('Experimental - Predicted $T_\mathrm{C}$ Error (K)')
    plt.title('2NN Predicted $T_\mathrm{C}$ vs Error')


def errorKNNReal():
    newMat = importFormattedData()
    knn = 2
    pred, real = fastKNNtrain(newMat, knn)

    temps = []

    errors = []
    b = []
    A = []
    for i in real:
        temps.append(i[1])


    for i in range(len(pred)):
        trueTC = temps[i]

        err = trueTC - (pred[i])
        errors.append(err)
        A.append([1, trueTC])
        b.append([err])



    overx = []
    overy = []
    underx = []
    undery = []

    for i in range(len(errors)):
        if errors[i] >= 0:
            underx.append(temps[i])
            undery.append(errors[i])
        else:
            overx.append(temps[i])
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
    plt.scatter(overx, overy, marker = '.', c = 'b', label = 'Model Overestimates')
    plt.scatter(underx, undery, marker = '.', c = 'r', label = 'Model Underestimates')
    plt.xlabel('Experimental $T_\mathrm{C}$ (K)')

    plt.legend(loc = 'lower right', markerscale = 3)
    plt.ylabel('Experimental - Predicted $T_\mathrm{C}$ Error (K)')
    plt.title('2NN Experimental $T_\mathrm{C}$ vs Error')


KNNpredict(2)
errorKNNPredicted()
errorKNNReal()

plt.show()