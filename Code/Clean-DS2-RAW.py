import numpy as np
import pandas as pd
import csv
import pymatgen.core.composition as pmg

# Reads in raw data and returns it in a list of lists
def import_data(): 
    data = []

    file = open('Data/DS2-RAW.txt', 'r')

    for line in file:
        ln = line.strip()
        ln = ln.split('\t')
        data.append(ln)

    for i in range(1, len(data)):
        for j in range(1,len(data[i])):
            ls = data[i][j].split(';')
            ls.pop(-1)
            data[i][j] = ls

    file.close()

    df = pd.DataFrame(data[1:], columns=['ChemicalFormula', 'ListOfElements', 'ListOfElementsQ', 'CurieTemperature'])

    print(df)

    return df

# Reads in raw data and returns it in a list of lists
def importOldData(): 
    data = []

    with open('Data/DS!-Incompatible.csv') as myFile:
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

def check_dupes():
    df = import_data()
    print(len(df['ChemicalFormula']))
   
    for i in df['ChemicalFormula']:
        #print(i)
        ct = 0
        for j in df['ChemicalFormula']:
            if i == j:
                ct += 1
        #print(ct)
        
        if ct > 1:
            print("THERE IS A DUPLICATE AMONG US")
            return

def find_poss_elements():
    df = import_data()
    elems = set()

    for i in df['ListOfElements']:
        for j in i:
            elems.add(j)

    elem_list = list(elems)
    elem_list.sort()

    old = importOldData()
    print(old[0][2::])

    print(len(elem_list))
    print(elem_list)

    all_elems = set()

    for i in old[0][2::]:
        all_elems.add(i)

    for i in elem_list:
        all_elems.add(i)
    
    all_elems_list = list(all_elems)
    all_elems_list.sort()

    print("Length of elems is: ", len(all_elems_list))
    print(all_elems_list)

df = import_data()
elements = ['Ag', 'Al', 'Am', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Bk', 'Br', 'C', 'Ca', 
            'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F',
              'Fe', 'Ga', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 'K', 'La',
                'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'Np', 'O', 'Os',
                  'P', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb',
                    'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'Ti', 'Tl',
                      'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']


#length is 87


def create_matrix():
    header = ['Name', 'TC'] + elements

    mat = [header]
    relErr = []

    for i in range(len(df['ChemicalFormula'])):
        if df['ChemicalFormula'][i] == 'AgAuZn2' or df['ChemicalFormula'][i] == 'N2OTh2':
            print("Found: ", df['ChemicalFormula'][i])
            continue
        row = [df['ChemicalFormula'][i]]
        elems = df['ListOfElements'][i]
        nums = df['ListOfElementsQ'][i]

        for j in range(len(nums)):
            nums[j] = float(nums[j])

        total = np.sum(nums)

        for j in range(len(nums)):
            nums[j] = nums[j]/total
        

        tc = df['CurieTemperature'][i]
 
        for j in range(len(tc)):
            
            tc[j] = float(tc[j].replace(' K', ''))
        

        rng = max(tc) - min(tc)
        avg = np.mean(tc)
        relErr.append(rng/avg)
        

        tc_median = np.median(tc)
        row.append(tc_median)

        for element in elements:
            found = False
            for k in range(len(elems)):

                if element == elems[k]:
                    row.append(nums[k])
                    found = True
            if not found:
                row.append(0.0)

        mat.append(row)

    relative_error = np.mean(relErr)
    print(len(relErr))
    print('The relative error is: ', relative_error)

    #print(mat)
    return(mat)

def save_data():
    d = create_matrix()
    file = open('./Data/DS2.csv', 'w')
    
    with file:
        write = csv.writer(file)
        write.writerows(d)


save_data()


