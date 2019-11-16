import numpy as np
import matplotlib.pyplot as plt 
import os, csv, matplotlib

datasetPath = "..\\Dataset\\artificial-new\\level0\\rebalanced\\dataset\\R11"
figurePath = "..\\Dataset\\artificial-new\\level0\\rebalanced\\base alpha histogram\\R11"

intervalMinVals = [0.05 * i for i in range(0, 21)]
finishedDatasetCount = 0
subPlotCount = 1
finishedImageCount = 0
rowCount = 4
columnCount = 5
matplotlib.rcParams.update({'font.size': 7})
filenames = os.listdir(datasetPath)
filenames.sort()

for filename in filenames: 
    with open(os.path.join(datasetPath, filename)) as file: 
        table = [row for row in csv.reader(file)][1:] 
    alphas = [table[r][-1] for r in range(len(table))]
    alphas = np.array(alphas).astype(np.float)
    
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.subplot(rowCount, columnCount, subPlotCount)
    plt.hist(alphas, bins = intervalMinVals)
    plt.title(filename.split('.')[0], loc = "left", pad = -13) 
    finishedDatasetCount += 1
    subPlotCount += 1

    if finishedDatasetCount == 524 or finishedDatasetCount == 706 or subPlotCount == rowCount * columnCount + 1: 
        plt.gca().axes.get_yaxis().set_ticks([])
        finishedImageCount += 1
        subPlotCount = 1
        imageFilename = f"R11_Hist{finishedImageCount}.png"
        plt.gcf().set_size_inches(11.7, 8.3)
        plt.savefig(os.path.join(figurePath, imageFilename), dpi = 300)
        print(f"Exported {imageFilename}")
        plt.clf()
