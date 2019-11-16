import numpy as np
import matplotlib.pyplot as plt 
import os, csv, matplotlib, math

matplotlib.rcParams.update({'font.size': 7})
graphTypes = { "origAlpha": 3, "knnAlpha": 6, "knnAShift": 7, "nbAlpha": 10, "nbAShift": 11, "dtAlpha": 14, "dtAShift": 15 }
transformations = ["original", "root", "logarithm", "exponential", "inverse"]
datasetTypes = ["LS-DT", "LS-KNN", "LS-NB", "RT"]
subplotIndex = 1

def FilterFilenamesStartWith(filenames, prefix):
    results = []
    for path in filenames:
        if os.path.basename(path).startswith(prefix):
            results.append(path)
    return results

def TransformValue(value, mode):
    if mode == "original":
        return value
    if mode == "root":
        return value ** 0.5
    if mode == "logarithm":
        return None if value == 0 else math.log(abs(value))
    if mode == "exponential":
        return math.exp(value)
    if mode == "inverse":
        return None if value == 0 else 1 / value

for graphType in graphTypes:
    for transformation in transformations:
        os.makedirs(f"..\\Dataset\\artificial-new\\level1\\level1-summary\\CategorizedHistogram-unfixedbin\\{graphType}\\{transformation}")
        plotIndex = 1
        # draw 800 histograms
        for datasetType in datasetTypes:
            for filename in os.listdir(f"..\\Dataset\\artificial-new\\level1\\level1-summary\\{datasetType}"):
                filename = f"..\\Dataset\\artificial-new\\level1\\level1-summary\\{datasetType}\\" + filename
                rows = []
                with open(filename) as file:
                    rows = list(file)
                valueCollection = []
                for row in rows[1:]:
                    value = float(row.split(',')[graphTypes[graphType]])
                    valueTransformed = TransformValue(value, transformation)
                    if valueTransformed is not None:
                        valueCollection.append(valueTransformed)
                valueCollection = np.array(valueCollection).astype(np.float)
                plt.gca().axes.get_yaxis().set_ticks([])
                plt.subplot(5, 4, subplotIndex)
                subplotIndex += 1
                plt.hist(valueCollection, bins = 20)
                plt.title(os.path.basename(filename)[:-9], loc = "left", pad = -13)
                if subplotIndex > 20:
                    graphName = f"{graphType}-{transformation}-{plotIndex}"
                    plt.suptitle(graphName, y = 0.94, fontsize = 15)
                    plt.gca().axes.get_yaxis().set_ticks([])
                    plt.gcf().set_size_inches(9, 8.3)
                    plt.savefig(f"..\\Dataset\\artificial-new\\level1\\level1-summary\\CategorizedHistogram-unfixedbin\\{graphType}\\{transformation}\\{graphName}.png", dpi = 300)
                    print(f"{graphName} saved. ")
                    plt.clf()
                    subplotIndex = 1
                    plotIndex += 1
