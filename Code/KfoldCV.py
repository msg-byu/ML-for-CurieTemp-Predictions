import numpy as np
import matplotlib.pyplot as plt
import csv
import statistics as sts
import random as rnd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
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



# Separates data into a training set and a test set
def selectData():

    lst = importData("Data/DS1.csv")

    # making dataframe 
    df = pd.DataFrame(lst[1:], columns=lst[0])
    ndf = df.loc[:, df.columns!='Name']
    

    y = ndf['TC']
    X = ndf.loc[:, ndf.columns!='TC']


    return X, y


def KfoldCrossVal():
    X, y = selectData()

    rcv = RepeatedKFold(n_splits=3, n_repeats=50, random_state=1)

    model = RandomForestRegressor(max_depth=90, n_estimators=1800, min_samples_leaf = 1, min_samples_split = 2, random_state=30)

    scores = cross_val_score(model, X, y, scoring = 'neg_mean_absolute_error', cv=rcv, n_jobs=-1)

    # report performance
    print('MAE: %.3f  STD:(%.3f)' % (np.abs(np.mean(scores)), np.std(scores)))        


KfoldCrossVal()