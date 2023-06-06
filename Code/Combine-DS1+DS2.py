import numpy as np
import csv
import matplotlib.pyplot as plt

# Reads in raw data and returns it in a list of lists
def importData(filename): 
    data = []

    #with open('Data/UpdatedCT.csv') as myFile:
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

    #print(data)
    return data


def countOverlap():
    sanvito = importData('Data/DS1.csv')
    valentin = importData('Data/DS2.csv')
    same_comps = 0
    thresh = 0.01

    for s in range(1,len(sanvito[1::])):
        for v in range(1,len(valentin[1::])):
            is_same = True
            
            for i in range(2, len(sanvito[s])):
                dif = np.abs(valentin[v][i] - sanvito[s][i])
                if dif > thresh:
                    is_same = False
            if is_same == True:
                same_comps += 1
                #print("Compounds are: ",sanvito[s][0],", ", valentin[v][0])
                break

    print("There are ", same_comps, " of the same compounds between the sets")

#countOverlap()

def overlapTCHistogram():
    sanvito = importData('Data/DS1.csv')
    valentin = importData('Data/DS2.csv')
    same_comps = 0
    thresh = 0.02
    tcDiffs = []

    for s in range(1,len(sanvito[1::])):
        for v in range(1,len(valentin[1::])):
            is_same = True
            
            for i in range(2, len(sanvito[s])):
                dif = np.abs(valentin[v][i] - sanvito[s][i])
                if dif > thresh:
                    is_same = False
            if is_same == True:
                same_comps += 1
                tcDiff = np.abs(sanvito[s][1] - valentin[v][1])
                tcDiffs.append(tcDiff)
                #print("Compounds are: ",sanvito[s][0],", ", valentin[v][0])
                break

    plt.hist(tcDiffs, bins=50)
    plt.xlabel("Tc Difference (0.02)")
    plt.ylabel("Number of compounds")
    plt.xlim(0,400)
    plt.show() 

    print("There are ", same_comps, " of the same compounds between the sets")

#overlapTCHistogram()

def createCombinedData():
    sanvito = importData('Data/DS1.csv')
    valentin = importData('Data/DS2.csv')
    thresh = 0.01

    combined = valentin.copy()

    for s in range(1,len(sanvito[1::])):
        for v in range(1,len(valentin[1::])):
            is_same = True
            for i in range(2, len(sanvito[s])):
                dif = np.abs(valentin[v][i] - sanvito[s][i])
                if dif > thresh:
                    is_same = False

            if is_same == True:
                print("Compounds are: ",sanvito[s][0],", ", valentin[v][0])
                break
        if is_same == False:
            combined.append(sanvito[s])

    print("Combined length: ", len(combined))
    return combined


def save_data():
    d = createCombinedData()
    file = open('./Data/DS1+DS2.csv', 'w')
    
    with file:
        write = csv.writer(file)
        write.writerows(d)

save_data()









