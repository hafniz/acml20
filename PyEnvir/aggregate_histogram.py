import numpy as np
import matplotlib.pyplot as plt 
import os, csv, matplotlib, math

def GetFilenamesStartWith(paths, prefix):
    filenames = []
    for path in paths:
        if os.path.basename(path).startswith(prefix):
            filenames.append(path)
    return filenames

matplotlib.rcParams.update({'font.size': 7})
colIndexes = { "origAlpha": 3, "knnAlpha": 6, "knnAShift": 7, "nbAlpha": 10, "nbAShift": 11, "dtAlpha": 14, "dtAShift": 15}
for datasetType in ["LS-DT", "LS-KNN", "LS-NB", "RT"]:
    datasetFolder = f"..\\Dataset\\artificial-new\\level1\\level1-summary\\{datasetType}"
    os.chdir(datasetFolder)
    smryFilenames = os.listdir(datasetFolder)
    for figCat in colIndexes:
        title = f"{datasetType}.{figCat}"
        plt.suptitle(title, y = 0.94, fontsize = 15)
        #intervalMinVals = [0.05 * i for i in range(21)] if figCat.endswith("Alpha") else [(0.1 * i) - 1 for i in range(21)]
        for l in range(2, 21, 2):
            level = "{:0>2d}".format(l)
            targetFilenames = GetFilenamesStartWith(smryFilenames, f"{datasetType[:2]}{level}")
            figValueCollection = []
            for filename in targetFilenames:
                with open(filename) as file:
                    rows = list(file)
                    for row in rows[1:]:
                        value = float(row.split(',')[colIndexes[figCat]])
                        figValueCollection.append(math.exp(value))
            figValueCollection = np.array(figValueCollection).astype(np.float)
            plt.gca().axes.get_yaxis().set_ticks([])
            plt.subplot(3, 4, l / 2)
            #plt.hist(figValueCollection, bins = intervalMinVals)
            plt.hist(figValueCollection, bins = 20)
            plt.title(f"  {level}", loc = "left", pad = -13)
        plt.gca().axes.get_yaxis().set_ticks([])
        plt.gcf().set_size_inches(11.7, 8.3)
        plt.savefig(f"..\\Dataset\\artificial-new\\level1\\level1-summary\\AggregateHistogram-unfixedbin\\exponential\\{title}.png", dpi = 300)
        print(f"{title} saved. ")
        plt.clf()
