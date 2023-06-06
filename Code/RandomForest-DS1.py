import numpy as np
import matplotlib.pyplot as plt
import csv
import statistics as sts
import random as rnd
# random forest for making predictions for regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# Reads in raw data and returns it in a list of lists
def importFormattedData(): 
    data = []

    with open('Data/DS1.csv') as myFile:
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

    #print(data)
    return data

#counts the percentage of materials with a curie temperature at or below the specified parameter
def countLowTC(limit):
    matrix = importFormattedData()
    count = 0
    for i in matrix[1:]:
        if i[1] <= limit:
            count += 1
    
    percentage = (count/len(matrix[1:])) * 100
    opp = 100 - percentage

    print(percentage, " percent of the materials have a TC at or below ", limit, "K")
    print(opp, " percent of the materials have a TC above ", limit, "K")




# Separates data into a training set and a test set
def selectData(data):
    rnd.seed(50) #97, 70, 50
    train = []
    #sampleSize = 767
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
    matrix = importFormattedData()
    control, test = selectData(matrix[1::])

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

def KCrossVal():
    from sklearn.model_selection import RandomizedSearchCV

    matrix = importFormattedData()
    control, test = selectData(matrix[1::])
    trainMat = []
    trainTc = []
    print('Imported Data')
    for i in control:
        trainTc.append(i[1])
        trainMat.append(i[2::])
    X = np.array(trainMat)
    y = np.array(trainTc)
    print('Made array')

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap}

    # define the model
    model = RandomForestRegressor()
    model_rand = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # fit the model on the whole dataset
    model_rand.fit(X, y)
    print('Printing the best parameters')
    print(model_rand.best_estimator_)

    return(model_rand.best_estimator_)

def refinedKCrossVal():
    from sklearn.model_selection import GridSearchCV

    matrix = importFormattedData()
    control, test = selectData(matrix[1::])
    trainMat = []
    trainTc = []
    print('Imported Data')
    for i in control:
        trainTc.append(i[1])
        trainMat.append(i[2::])
    X = np.array(trainMat)
    y = np.array(trainTc)
    print('Made array')

    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': [90, 100, 110,120],
        'max_features': ['auto'],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2,4,6],
        'n_estimators': [1400, 1500, 1700, 1800]
        }

    # define the model
    model = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    # fit the model on the whole dataset
    grid_search.fit(X, y)
    print('Printing the best parameters')
    print(grid_search.best_params_)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = np.mean(errors)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

def getSets():
    matrix = importFormattedData()
    control, test = selectData(matrix[1::])
    trainMat = []
    trainTc = []
    print('Imported Data')
    for i in control:
        trainTc.append(i[1])
        trainMat.append(i[2::])
    X = np.array(trainMat)
    y = np.array(trainTc)
    print('Made array')
    testMat = []
    testTc = []
    for i in test:
        testTc.append(i[1])
        testMat.append(i[2::])

    base_model = RandomForestRegressor()
    base_model.fit(X, y)
    base_accuracy = evaluate(base_model, testMat, testTc)

    best_random = KCrossVal()
    random_accuracy = evaluate(best_random, testMat, testTc)

    print('Improvement of {:0.2f}%.'.format((random_accuracy - base_accuracy)))


def calcPlotErrorAbove600():
    pred, real = randForest()
    b50 = 0
    b100 = 0
    errors = 0
    count = 0 

    new_pred = []
    for i in pred:
        ad = (0.17175448 * i) - 38.14641777
        new_pred.append(i + ad)
    
    pred = new_pred



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


    print('Random Forest finished with a ',MAE,' kelvin mean absolute error')
    print('Random Forest finished with a ',r2,' R^2 score')

    plt.figure(figsize=(7.5, 7.5))
    plt.rcParams.update({'font.size': 18})
    plt.scatter(pred,real,marker=".")
    plt.plot([0,1400],[0,1400],color='red')
    plt.xlabel('Predicted $T_\mathrm{C}$ (K)')
    plt.ylabel('Experimental $T_\mathrm{C}$ (K)')
    plt.title('Random Forest Prediction')
    #plt.text(100, 1400, f'%d Kelvin Mean Average Error' % MAE, fontsize = 12)
    plt.text(100, 1400, f'%d \% within 50 K' % b50, fontsize = 17)
    plt.text(100, 1300, f'%d \% within 100 K' % b100, fontsize = 17)



    plt.show()


def errorHistogram():
    pred, real = randForest()
    errors = []

    for i in range(len(pred)):
        trueTC = real[i]

        err = trueTC - pred[i] 
        errors.append(err)
        
    

    n_bins = 30

    # We can set the number of bins with the *bins* keyword argument.
    plt.hist(errors, bins=n_bins)
    plt.xlabel('Real vs. Predicted Error')
    plt.title('Random Forest Error Histogram')

    plt.show()


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


calcPlotError()
errorTCPredicted()
errorTCReal()

plt.show()