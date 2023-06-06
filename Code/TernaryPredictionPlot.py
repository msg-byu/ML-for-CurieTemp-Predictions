import numpy as np
import matplotlib.pyplot as plt
import csv
import random as rnd
import plotly.figure_factory as ff

STEPSIZE = 10
STAT = 'max'



# MAKE SURE TO PUT REAL ELEMENTS BEFORE X
# 'XXX' For variable
ELEMENT1 = 'Co'
# 'XX' For variable
ELEMENT2 = 'Fe'
# 'X' For variable
ELEMENT3 = 'X'


IMPORT_FILENAME = 'Data/Generated Materials/GC_Ternary_' + ELEMENT1 + '+' + ELEMENT2 + '+' + ELEMENT3 + '.csv'
#SAVE_FILENAME =

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

    #print(data[0])
    return data

def splitName(name):
    splitted = []

    split1 = name.split('+')

    for i in split1:
        split2 = i.split(':')
        splitted.append([split2[0], float(split2[1])])

    return splitted

def joinName(compounds):
    
    name = ELEMENT1 + ':' + str(compounds[0][1]) + '+' + ELEMENT2 + ':' + str(compounds[1][1]) + '+' + ELEMENT3 + ':' + str(compounds[2][1])

    return name


def createTernaryCoordinates():

    data = importData(IMPORT_FILENAME)
    header = data[0]
    dict = {}
    middle = []
    
    for i in range(len(data)):
        if i == 0:
            continue
        compound = data[i][0]

        magnet = splitName(compound)
        
        if round(float(magnet[0][1]) * 100) == 40 and round(float(magnet[1][1]) * 100) == 25 and round(float(magnet[2][1]) * 100) == 35:
            middle.append([data[i][0], data[i][1]])


        name = joinName(splitName(compound))

        if name in dict.keys():
            dict[name].append(data[i][1])
        else:
            dict[name] = [data[i][1]]

    print('Dictionary length: ', len(dict))


    middle.sort(key = lambda x: x[1])
    print(middle)


    for comp in dict:
        #Choose Max or Average
        if STAT == 'max':
            mx = round(np.max(dict[comp]))
            dict[comp] = mx

        elif STAT == 'avg':
            avg = round(np.mean(dict[comp]))
            dict[comp] = avg


    e1Arr = []
    e2Arr = []
    e3Arr = []
    tcArr = []

    for comp in dict:
        composition = []
        key = comp

        compoundList = splitName(comp)

        for i in compoundList:
            composition.append([i[0], round(float(i[1]) * 100)])
            #composition.append([i[0], i[1]])

        e1Arr.append(composition[0][1])
        e2Arr.append(composition[1][1])
        e3Arr.append(composition[2][1])

        tcArr.append(dict[comp])

    print(e1Arr[0])
    print(e2Arr[0])
    print(e3Arr[0])
    print(tcArr[0])

    return np.array(e1Arr), np.array(e2Arr), np.array(e3Arr), np.array(tcArr)




def pythonTernary():
    e1,e2,e3,tc = createTernaryCoordinates()
    
    import ternary

    d = {}

    for i in range(len(e1)):
        d[(e1[i],e2[i],e3[i])] = tc[i]


    scale = 100
    figure, tax = ternary.figure(scale=scale)
    #figure.set_size_inches(10, 8)
    #vmin vmax

    

    tax.heatmap(d, style="t", cmap='hot')
    tax.boundary()

    figure.text(0.78, 0.9, "$T_\mathrm{C}$",fontsize=15)



    fontsize = 15
    offset = 0.2

    tax.ticks(axis='lbr', multiple=STEPSIZE, offset=0.02)
 
    

    if ELEMENT1 == 'XXX':
        lbl1 = "$X_\mathrm{1}$"
    else:
        lbl1 = ELEMENT1
    
    if ELEMENT2 == 'XX':
        if ELEMENT1 == 'XXX':
            lbl2 = "$X_\mathrm{2}$"
        else:
            lbl2 = "$X_\mathrm{1}$"
    else:
        lbl2 = ELEMENT2

    if ELEMENT3 == 'X':
        if ELEMENT2 == 'XX':
            if ELEMENT1 == 'XXX':
                lbl3 = "$X_\mathrm{3}$"
            else:
                lbl3 = "$X_\mathrm{2}$"
        else:
            lbl3 = "$X_\mathrm{1}$"
    else:
        lbl3 = ELEMENT3


    tax.left_axis_label('% ' + lbl3, fontsize=fontsize, offset=offset)
    tax.right_axis_label('% ' + lbl2, fontsize=fontsize, offset=offset)
    tax.bottom_axis_label('% ' + lbl1, fontsize=fontsize, offset=offset)

    
    tax.set_title(lbl1 + " + " + lbl2+ " + " + lbl3 + " $T_\mathrm{C}$ Heatmap", fontsize = fontsize, pad = 25)

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    tax._redraw_labels()
    #tax.savefig(SAVE_FILENAME)

    plt.show()

pythonTernary()

