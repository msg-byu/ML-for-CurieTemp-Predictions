import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pylab as plt
import seaborn as sns
import pandas as pd
import csv
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

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

def organizeData():
    mat = importFormattedData()

    TC = []
    design = []
    name = []

    for i in mat[1:]:
        TC.append(i[1])
        design.append(i[2:])
        name.append(i[0])


    X = np.array(design)
    y = np.array(TC)
    head = mat[0][2:]

    return X,y,head, name


X,y,head,name = organizeData()

def findCompound():
    for i in range(len(name)):
        if name[i] == 'N2OTh2':
            print(X[i])
            print(y[i])
            


MACHINE_EPSILON = np.finfo(np.double).eps
n_components = 2
perplexity = 30

def componentY():
    major = []
    myset = set()
    for i in X:
        max = 0
        for j in range(len(i)):
            if max < i[j]:
                max = i[j]
                index = j
        major.append(head[index])
        myset.add(head[index])
    
    siz = len(myset)

    return major, siz
        
majY,siz = componentY()


def easyTSNE2():
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    print("Finished TC")
    z = tsne.fit_transform(X)
    print("Finished fit")
    df = pd.DataFrame()
    sz = len(y)
    print(sz)
    df["Y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    lockx = z[:,0]
    locky = z[:,1]

    minx = -20
    maxx = 20
    miny = -80
    maxy = -50

    for i in range(len(lockx)):
        if lockx[i] > minx and lockx[i] < maxx:
            if locky[i] > miny and locky[i] < maxy:
                print(y[i])

    print("Made dataframe")
    sns.set(rc = {'figure.figsize':(7,7)},font_scale = 1.5)
    gfg = sns.scatterplot(x="comp-1", y="comp-2", hue=y,
    palette=sns.color_palette('turbo',as_cmap = True),
    data=df)
    gfg.set(title="TC data T-SNE projection")
    gfg.legend(fontsize = 16, title = "TC (Kelvin)", bbox_to_anchor=(1,1), loc='upper left')
    #gfg.legend(ncol=3, bbox_to_anchor=(0.90, 1.13), loc='upper left', borderaxespad=0,fontsize = 12)
    #plt.savefig('TCTSNE.png',bbox_inches='tight')
    plt.show()


def easyTSNE2Sorted():
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    print("Finished TC")
    z = tsne.fit_transform(X)
    print("Finished fit")
    df = pd.DataFrame()
    sz = len(y)

    sorter = []
    pca1 = z[:,0]
    pca2 = z[:,1]

    for i in range(len(y)):
        sorter.append([pca1[i], pca2[i], y[i]])

    sorter.sort(key = lambda x: x[2])

    p1 = []
    p2 = []
    tc = []
    for i in range(len(y)):
        p1.append(sorter[i][0])
        p2.append(sorter[i][1])
        tc.append(sorter[i][2])

    print(sz)
    df["Y"] = tc
    df["Comp-1"] = p1
    df["Comp-2"] = p2
    print("Made dataframe")
    sns.set(rc = {'figure.figsize':(8.5,7)},font_scale = 1.5)
    gfg = sns.scatterplot(x="Comp-1", y="Comp-2", hue=tc,
    palette=sns.color_palette('turbo', as_cmap = True),
    data=df)
    
    norm = plt.Normalize(min(tc), max(tc))
    sm = plt.cm.ScalarMappable(cmap= 'turbo', norm= norm)
    sm.set_array([])

    gfg.get_legend().remove()
    gfg.figure.colorbar(sm,  shrink = 0.7, aspect = 40, label = "$T_\mathrm{C}$ (Kelvin)" )
    #gfg.text(-77, 57, '(b)', fontsize = 18, weight='bold')
    gfg.set(title="$T_\mathrm{C}$ Data T-SNE Projection - DS1")
    #gfg.legend(fontsize = 16, title = "TC (Kelvin)", bbox_to_anchor=(1,1), loc='upper left')
    #gfg.legend(ncol=3, bbox_to_anchor=(0.90, 1.13), loc='upper left', borderaxespad=0,fontsize = 12)
    #plt.savefig('./Curie Temp Plots/Final Figs/TCTSNE.png',bbox_inches='tight')
    #plt.savefig('./Curie Temp Plots/Final Figs/Valentin-TCTSNE.png', bbox_inches='tight')



def tsnePinpointHighTC(xmin, xmax, ymin, ymax):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    print("Finished TC")
    z = tsne.fit_transform(X)
    print("Finished fit")
    df = pd.DataFrame()
    sz = len(y)

    sorter = []
    pca1 = z[:,0]
    pca2 = z[:,1]

    for i in range(len(y)):
        sorter.append([pca1[i], pca2[i], y[i], name[i]])

    sorter.sort(key = lambda x: x[2])

    p1 = []
    p2 = []
    tc = []
    comp = []
    for i in range(len(y)):
        p1.append(sorter[i][0])
        p2.append(sorter[i][1])
        tc.append(sorter[i][2])
        comp.append(sorter[i][3])

    print(sz)

    region = []

    for i in range(len(tc)):
        if p1[i] > xmin and p1[i] < xmax:
            if p2[i] > ymin and p2[i] < ymax:
                if tc[i] > 600:
                    region.append([p1[i], p2[i], tc[i], comp[i]])

    region.sort(key = lambda x: x[2], reverse=True)

    print(region)

    df["Y"] = tc
    df["Comp-1"] = p1
    df["Comp-2"] = p2
    print("Made dataframe")
    sns.set(rc = {'figure.figsize':(7,7)},font_scale = 1.5)
    gfg = sns.scatterplot(x="Comp-1", y="Comp-2", hue=tc,
    palette=sns.color_palette('turbo',as_cmap = True),
    data=df)
    gfg = annotateDS2(gfg)
    gfg.text(-57, 50, '(b)', fontsize = 18, weight='bold')
    gfg.set(title="$T_\mathrm{C}$ Data T-SNE Projection - DS1")
    gfg.legend(fontsize = 16, title = "$T_\mathrm{C}$ (Kelvin)", bbox_to_anchor=(1,1), loc='upper left')
    #gfg.legend(ncol=3, bbox_to_anchor=(0.90, 1.13), loc='upper left', borderaxespad=0,fontsize = 12)
    #plt.savefig('./Curie Temp Plots/Final Figs/TCTSNE.png',bbox_inches='tight')
    #plt.savefig('./Curie Temp Plots/Final Figs/Valentin-TCTSNE.png', bbox_inches='tight')
    plt.show()

#Does TSNE projection and return x coordinate array and y coordinate array
def easyTSNE():
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    print("Finished TC")
    z = tsne.fit_transform(X)
    print("Finished fit")
    df = pd.DataFrame()
    sz = len(majY)
    print(sz)
    df["majY"] = majY
    print(head)

    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    print("Made dataframe")

    return z[:,0], z[:,1]

#Grab TSNE x and y coordinates
tx,ty = easyTSNE()


# Plots the TSNE data as pie charts for each point. 
# Takes x coordinate and y coordinate arrays
def pieParty(xcor, ycor):
    import matplotlib as mpl

    #Create custom color map

    #Choose how many colors you need
    num_colors = len(head)

    #Choose colormap
    cm = mpl.cm.get_cmap(name='jet')

    #Divide color map into the ampunt of colors you need and save them in an array
    colors = [cm(1.*i/num_colors) for i in range(num_colors)]

    
    colorCopy = colors.copy()

    #Indexes of Co, Fe, Ni, O
    ind = [18, 27, 39, 51]

    #Swap colors of major elements
    colors[ind[1]] = colorCopy[-1]
    colors[-1] = colorCopy[ind[1]]

    colors[ind[3]] = colorCopy[66]
    colors[66] = colorCopy[ind[3]]



    pies = []

    # Iterates through design matrix
    for row in X:
        #print(row)
        pie = []

        # Round every value in the row to 2 decimal places
        row = np.round(row,2)
        tot = 0
        prev = 0

        # Iterate through values in the row
        for i in range(len(row)):

            # Skip the value if it's 0 and keep going
            if row[i] > 0:

                # Keeps track of the total percentage of the pie so we know where to start each piece
                tot = tot + row[i]

                # I don't remember exactly how this works but it creates the pie piece and saves it as xy1
                x1 = np.cos(2 * np.pi * np.linspace(prev, tot))
                y1 = np.sin(2 * np.pi * np.linspace(prev, tot))
                xy1 = np.row_stack([[0, 0], np.column_stack([x1, y1])])

                # This has something to do with the size of the piece
                s1 = np.abs(xy1).max()


                #Appends this piece as a list to the current pie 
                # s1 - something to do with the size of the piece
                # xy1 - array representing pie piece
                # i - feature position so we know what color it should correspond to
                pie.append([s1, xy1, i])

                prev = tot
        

        # Append completed pie as a 2D list to the pies list
        pies.append(pie)


    # Set up plot format
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlabel('Comp-1', fontsize = 18)
    ax.set_ylabel('Comp-2', fontsize = 18)
    #ax.text(-65, 72, '(a)', fontsize = 18, weight='bold')
    #ax.text(-77, 60, '(a)', fontsize = 18, weight='bold')
    ax.set_title("$T_\mathrm{C}$ Data T-SNE Projection - DS1", fontsize = 18)

    # size used to scale the size of the pie markers
    size = 20
    ind = 0
    k = 0

    # loops through the list of pies
    for i in pies:
        # loops through each pie
        for j in i: 

            # Plot each pie piece individually. xcor[k] and ycor[k] are the coordinates for the actual data point
            # marker is the custom pie slice marker we made
            # s is the size of the marker
            # colors[j[2]] grabs the corresponding color in the custom colors array we made earlier
            ax.scatter(xcor[k], ycor[k], marker=j[1], s=j[0]**2 * size, facecolor=colors[j[2]])

            ind += 1
        k += 1

    # Just a bunch of stuff the format the legend
    legend_elements = []
    for i in range(len(head)):
        legend_elements.append(mpl.lines.Line2D([0], [0], marker='o', linestyle='None', label=head[i],
                          color=colors[i], markersize=6))

    legend = ax.legend(handles = legend_elements, ncol=4, bbox_to_anchor=(1,1.05), loc='upper left', borderaxespad=0, fontsize = 14, columnspacing = 0.25)

    # Plot circles around chosen clusters
    #circleRegionsMBP(ax)

    # plot window is too small to fit the legend so save the plot directly
    #plt.savefig('./Curie Temp Plots/Final Figs/Valentin-ELEMTSNEPIE.png', bbox_inches='tight')
    #plt.savefig('./Curie Temp Plots/Final Figs/ELEMTSNEPIE.png', bbox_inches='tight')
    

easyTSNE2Sorted()
pieParty(tx, ty)


plt.show()







