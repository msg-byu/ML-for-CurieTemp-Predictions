import numpy as np
import matplotlib.pyplot as plt
import csv
import statistics as sts
import random as rnd

# Reads in raw data and returns it in a list of lists
def importData(): 
    data = []

    with open('Data/DS1-RAW.csv') as myFile:
        csvdata = csv.reader(myFile, delimiter = ',')

        for i in csvdata:
            data.append(i)

    myFile.close()

    return data

# Returns a list matrix containing rows with the compound name, Curie temperature, and composition
def CreateMatrix(data):
    length = 86
    header = ['Name', 'Temp', 'Ag', 'Al', 'Am', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Bk', 'Br',
    'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 
    'Eu', 'F', 'Fe', 'Ga', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 
    'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'Np', 
    'O', 'Os', 'P', 'Pb', 'Pd', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Ru', 'S', 
    'Sb', 'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'Ti', 
    'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']
    matrix = [header]
    for i in range(1,len(data)):
        tarr = np.zeros(len(header))
        temp = tarr.tolist()
        temp[1] = data[i][2]
        temp[0] = data[i][1]
 
        elem = []
        
        name = data[i][1]
        j = 0
        while j < len(name): # Iterates through each compound name and finds how much of each element is in a compound
            while name[j].isalpha():
                elem.append(name[j])
                j += 1
            str = "".join(elem)

            for k in range(length):
                if str == header[k]:
                    index = k
            elem = []

            while name[j].isnumeric():
                elem.append(name[j])
                j += 1
                if j == len(name):
                    break
            str = "".join(elem)
            num = int(str)
            temp[index] = num
            elem = []   
        matrix.append(temp)
    return matrix

def fracConvert(matrix):
    for i in range(1,len(matrix)):
        row = matrix[i]
        total = sum(row[2::])
        for j in range(len(row)):
            if j > 2:
                matrix[i][j] = row[j]/total
    return matrix

# Finds duplicate compounds, takes the median of their Curie temperatures, and returns the refined matrix
def siftData(myMatrix): 
    names = []
    length = len(myMatrix)
    newMat = [myMatrix[0]]
    for i in range(1,length):
        if myMatrix[i][0] in names:
            continue
        else:
            names.append(myMatrix[i][0])
        temps = []
        for j in range(1,length):
            if myMatrix[i][0] == myMatrix[j][0]:
                temps.append(float(myMatrix[j][1]))
        med = sts.median(temps)
        row = myMatrix[i]
        row[1] = med
        newMat.append(row)
    return newMat

def addTC0(matrix):
    ferros = ['Co', 'Fe', 'Ni', 'Gd', 'Tb', 'Dy', 'Nd', 'Tm', 'Er', 'Ho', 'Pr']
    header = matrix[0]
    length = len(header)
    for i in range(length):
        boo = False
        row = np.zeros(length)
        row = row.tolist()
        for j in ferros:
            if header[i] == j:
                boo = True
                break
        if boo == False:
            row[0] = header[i]
            row[1] = 0
            row[i] = 1
            matrix.append(row)

    print(matrix)
    print(len(matrix))

    return matrix

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

    if dist < 0.01 and dist != 0:
        print('Distance = ',dist)

    return dist


elements = ['Ag', 'Al', 'Am', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Bk', 'Br', 'C', 'Ca', 
            'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F',
              'Fe', 'Ga', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 'K', 'La',
                'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'Np', 'O', 'Os',
                  'P', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb',
                    'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'Ti', 'Tl',
                      'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']


#length is 87

header = ['Name', 'TC'] + elements

mat = CreateMatrix(importData())
nmat = siftData(mat)
nmat = fracConvert(nmat)
nmat = addTC0(nmat)

old_data = nmat
old_header = old_data[0]

print(header)

def rebuild_old_matrix():
    mat = []
    mat.append(header)
    for row in old_data[1::]:
        old_comp = row[2::]
        new_comp = [row[0], row[1]]

        for elem in elements:
            found = False
            for i in range(len(old_comp)):
                if elem == old_header[i + 2]:
                    new_comp.append(old_comp[i])
                    found = True
            if not found:
                new_comp.append(0.0)
        
        mat.append(new_comp)
    return mat

def save_data():
    d = rebuild_old_matrix()
    file = open('DS1.csv', 'w')
    
    with file:   
        write = csv.writer(file)
        write.writerows(d)



save_data()




#Save nmat