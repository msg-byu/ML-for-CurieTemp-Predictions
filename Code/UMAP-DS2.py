import csv
import umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

# Reads in raw data and returns it in a list of lists
def importFormattedData(): 
    data = []

    with open('Data/DS2.csv') as myFile:
    
        csvdata = csv.reader(myFile, delimiter = ',')

        for i in csvdata:
            data.append(i)

    myFile.close()

    lend = len(data)
    lenr = len(data[0])

    head = data[0]

    for row in range(lend):
        if row == 0:
            continue
        for ele in range(lenr):
            if ele > 0:
                data[row][ele] = float(data[row][ele])

     # making dataframe 
    df = pd.DataFrame(data[1:], columns=data[0])
    ndf = df.loc[:, df.columns!='Name']

    y = ndf['TC']
    X = ndf.loc[:, ndf.columns!='TC']

    #print(data)
    return X, y, head

design, target, header = importFormattedData()


#Finds the majority element in each compound
def componentY():
    X = design.values.tolist()
    major = []
    head = header[2:]
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

def umapProjection():

    X = design.copy()
    y = target.copy()
    manifold = umap.UMAP(random_state=42).fit(X, y)
    X_reduced = manifold.transform(X)

    print(X_reduced)

    #plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=0.5)
   

    return X_reduced, y


def colorByTc(X_reduced, y):

    sorter = []
    pca1 = X_reduced[:, 0]
    pca2 = X_reduced[:, 1]

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


    df = pd.DataFrame()
    df["Y"] = tc
    df["Comp-1"] = p1
    df["Comp-2"] = p2
    print("Made dataframe")
    sns.set(rc = {'figure.figsize':(8.5,7)},font_scale = 1.5)
    gfg = sns.scatterplot(x="Comp-1", y="Comp-2", hue=tc,
    palette=sns.color_palette('turbo', as_cmap = True),
    data=df)
    
    norm = plt.Normalize(min(y), max(y))
    sm = plt.cm.ScalarMappable(cmap= 'turbo', norm= norm)
    sm.set_array([])

    gfg.get_legend().remove()
    gfg.figure.colorbar(sm,  shrink = 0.7, aspect = 40, label = "$T_\mathrm{C}$ (Kelvin)" )

    gfg.set(title="$T_\mathrm{C}$ Data UMAP Projection - DS2")

    plt.savefig('./Plots/UMAP-Temp-DS2.png', bbox_inches='tight')


def colorByComposition(X_reduced, y):

    tx = X_reduced[:, 0]
    ty = X_reduced[:, 1]

    import matplotlib as mpl
    head = header[2:]

    #Choose how many colors you need
    num_colors = len(head)

    #Choose colormap
    cm = mpl.cm.get_cmap(name='jet')

    #Divide color map into the ampunt of colors you need and save them in an array
    colors = [cm(1.*i/num_colors) for i in range(num_colors)]

    
    colorCopy = colors.copy()

    #Indexes of Co, Fe, Ni, O
    ind = [17, 26, 38, 50]

    #Swap colors of major elements
    colors[ind[1]] = colorCopy[-1]
    colors[-1] = colorCopy[ind[1]]

    colors[ind[3]] = colorCopy[66]
    colors[66] = colorCopy[ind[3]]



    # Set up plot format
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlabel('Comp-1', fontsize = 18)
    ax.set_ylabel('Comp-2', fontsize = 18)

    
    
    #ax.text(-77, 60, '(a)', fontsize = 18, weight='bold')
    ax.set_title("$T_\mathrm{C}$ Data UMAP Projection - DS2", fontsize = 18)

    # loops through the list of points
    for i in range(len(tx)):
        # loops through each pie
        color_ind = head.index(majY[i])
        #print(color_ind)
        # Plot each pie piece individually. xcor[k] and ycor[k] are the coordinates for the actual data point
        # marker is the custom pie slice marker we made
        # s is the size of the marker
        # colors[j[2]] grabs the corresponding color in the custom colors array we made earlier
        ax.scatter(tx[i], ty[i], facecolor=colors[color_ind], edgecolors='black', linewidth=0.4)

    # Just a bunch of stuff the format the legend
    legend_elements = []
    for i in range(len(head)):
        legend_elements.append(mpl.lines.Line2D([0], [0], marker='o', linestyle='None', label=head[i],
                          color=colors[i], markersize=6))
    #circleRegionsMBP(ax)

    legend = ax.legend(handles = legend_elements, ncol=4, bbox_to_anchor=(1,1.05), loc='upper left', borderaxespad=0, fontsize = 14, columnspacing = 0.25)


    # plot window is too small to fit the legend so save the plot directly
    #plt.savefig('./Curie Temp Plots/Final Figs/Valentin-ELEMTSNEPIE.png', bbox_inches='tight')
    plt.savefig('./Plots/UMAP-ELEM-DS2.png', bbox_inches='tight')
    



xmat,ymat = umapProjection()
colorByTc(xmat,ymat)
colorByComposition(xmat, ymat)

plt.show()





