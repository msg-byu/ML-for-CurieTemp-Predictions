import numpy as np
import matplotlib.pyplot as plt
import csv
import random as rnd
from sklearn.ensemble import RandomForestRegressor



#PARAMETERS

STEPSIZE = 1

# 'XXX' For variable
ELEMENT1 = 'Co'
# 'XX' For variable
ELEMENT2 = 'Fe'
# 'X' For variable
ELEMENT3 = 'X'

#Elements to hold out
EXCLUDED_ELEMENTS = []

TRAINING_DATA_FILENAME = 'Data/DS1+DS2.csv'

SAVE_FILENAME = 'GC_Ternary_' + ELEMENT1 + '+' + ELEMENT2 + '+' + ELEMENT3 + '.csv'
#SAVE_FILENAME = 'GC_Ternary_' + ELEMENT1 + '80' + '+' + ELEMENT2 + '+' + ELEMENT3 + '.csv'


HEADER = ['Name', 'Temp', 'Ag', 'Al', 'Am', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Bk', 'Br', 'C', 'Ca', 
            'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F',
              'Fe', 'Ga', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 'K', 'La',
                'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'Np', 'O', 'Os',
                  'P', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb',
                    'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'Ti', 'Tl',
                      'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

#print(header)


#Use findBestCompounds to find the values for best_elems, required_elems, required_elems_indexes, and indexes
BEST_ELEMENTS = ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 
'C', 'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cu', 
'Dy', 'Er', 'Eu', 'F', 'Fe', 'Ga', 'Gd', 'Ge', 'H', 
'Hf', 'Ho', 'In', 'Ir', 'K', 'La', 'Li', 'Lu', 'Mg', 
'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'O', 'Os', 
'P', 'Pd', 'Pr', 'Pt', 'Pu', 'Re', 'Rh', 'Ru', 'S', 
'Sb', 'Sc', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Th', 
'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

print(len(BEST_ELEMENTS))

def getHeaderIndex(elem):
    for i in range(len(HEADER)):
        if elem == HEADER[i]:
            return i
    return "ERROR"


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

    return data


TRAINING_DATA = importData(TRAINING_DATA_FILENAME)




def generateData():
    print("Generating Data...")
    percentages = getPercentages()


    data = [HEADER]

    print("Generating compounds with " + ELEMENT1 + ", " + ELEMENT2 + ", " + ELEMENT3)
    compounds = getCompounds()
    length = len(compounds)
    iter = len(compounds)

    for compound in compounds:
        print("Compound combinations left: ", iter, "/", length)
        iter -= 1
        for percentage in percentages:
            row = createTernary(compound, percentage)
            data.append(row)

    
    return data

        
def getCompounds():
    print("Creating all possible compounds...")
    compounds = []

    e1 = None
    e2 = None
    e3 = None

    if ELEMENT1 == 'XXX':
        e1 = BEST_ELEMENTS.copy()
    else:
        e1 = [ELEMENT1]

    if ELEMENT2 == 'XX':
        e2 = BEST_ELEMENTS.copy()
    else:
        e2 = [ELEMENT2]

    if ELEMENT3 == 'X':
        e3 = BEST_ELEMENTS.copy()
    else:
        e3 = [ELEMENT3]

    for i in e1:
        if i in EXCLUDED_ELEMENTS: continue
        for j in e2:
            if j in EXCLUDED_ELEMENTS: continue
            if i == j:
                continue
            for k in e3:
                if k in EXCLUDED_ELEMENTS: continue
                if i == k or j == k:
                    continue

                compound = [i, j, k]
                compounds.append(compound)

    
    print("There are ",len(compounds), " compound combinations")
    return compounds



def getPercentages():
    print("Creating all possible percentages...")
    if (100 % STEPSIZE) != 0:
        print("CHOOSE A FACTOR OF 100 FOR THE STEP SIZE")
        return
    

    percentages = []
    increments = range(0, (100 + STEPSIZE), STEPSIZE)
    #increments = range(80, (100 + STEPSIZE), STEPSIZE)


    for i in increments:
        perc1 = i/100
        for j in range(0, ((100 - i) + STEPSIZE), STEPSIZE):
            perc2 = j/100
            perc3 = (100 - i - j)/100

            percentages.append([perc1, perc2, perc3])


   
    print("There are ",len(percentages), " percentages")
    #print(percentages)
    return percentages



def createTernary(elements, percentages):
    row = np.zeros(len(HEADER))
    row = row.tolist()
    #name = elements[0] + str(percentages[0]) + elements[1] + str(percentages[1]) + elements[2] + str(percentages[2])
    name = elements[0] + ':' + str(percentages[0]) + '+' + elements[1]  + ':' + str(percentages[1]) + '+' + elements[2] + ':' + str(percentages[2])
    row[0] = name

    ind1 = getHeaderIndex(elements[0])
    ind2 = getHeaderIndex(elements[1])
    ind3 = getHeaderIndex(elements[2])

    row[ind1] = percentages[0]
    row[ind2] = percentages[1]
    row[ind3] = percentages[2]

    return row


# Separates data into a training set and a test set
def selectRFData():
    train = TRAINING_DATA
    train = filterTC(train[1:])
    test = generateData()

    print('Training set size = ', len(train))
    print('Test set size = ', len(test))

    

    return train, test[1:]

def randForest():
    
    train, test = selectRFData()

    

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
    print("Training model...")
    model.fit(X, y)
    
    from pprint import pprint
    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(model.get_params())

    testMat = []
    for i in test:
        testMat.append(i[2::])
    # make predictions
    print("Making predicitons...")
    yhat = model.predict(testMat)
    
    return yhat, test


def sortPredictions():
    print("Sorting data by Curie temperature...")
    predictions, genData = randForest()

    for i in range(len(genData)):
        genData[i][1] = round(predictions[i])

    genData.sort(key = lambda x: x[1], reverse= True)

    genData.insert(0,HEADER)

    
    return genData

def filterTC(data):
    newdata = []
    for i in data:
        if i[1] > 600:
            newdata.append(i)

    print(len(newdata), " compounds above 600K")

    return newdata


def prepData():

    data = sortPredictions()
    names = []
    
    for i in range(len(data)):
        if i == 0:
            continue
        compound = []
        for j in range(2, len(data[i])):
            if data[i][j] > 0:
                compound.append([HEADER[j], data[i][j]])

        compound.sort()
        name = ""
        for k in compound:
            name = name + k[0] + str(k[1])

        #data[i][0] = name
        names.append(name)

    return names, data

def removeDuplicates():
    print('Removing duplicate compounds...')
    dict = {}
    names, data = prepData()

    newData = [HEADER]

    for i in range(0, len(names)):
        if names[i] in dict.keys():
            continue
        else:
            dict[names[i]] = 1
            newData.append(data[i + 1])


    print(len(data))
    print(len(newData))

    return newData

def save_data():
    
    d = removeDuplicates()
    
    print("Saving File...")
    filepath = './Data/Generated Materials/' + SAVE_FILENAME

    file = open(filepath, 'w')

    print("Number of compounds in generated data: ", len(d[1:]))
    
    with file:
        write = csv.writer(file)
        write.writerows(d)


removeDuplicates()










        
