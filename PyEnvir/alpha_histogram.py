import numpy as np
import matplotlib.pyplot as plt 
import os, csv, matplotlib

datasetPath = "..\\Dataset\\artificial-R\\dataset"
figurePath = "..\\Dataset\\artificial-R\\base alpha histogram"

filenames = []
for root, dirs, files in os.walk(datasetPath):
    for filename in files:
        filenames.append(os.path.join(root, filename))

intervalMinVals = [0.05 * i for i in range(0, 21)]
matplotlib.rcParams.update({'font.size': 7})

for filename in filenames: 
    table = []
    with open(filename) as file: 
        table = [row for row in csv.reader(file)][1:] 
    alphas = [table[r][-1] for r in range(len(table))]
    alphas = np.array(alphas).astype(np.float)
    
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.hist(alphas, bins = intervalMinVals)
    basename = os.path.basename(filename).split('.')[0]
    plt.title(basename) 
    plt.gcf().set_size_inches(5.83, 4.13)
    plt.savefig(os.path.join(figurePath, f"{basename}.png"), dpi = 300)
    print(f"Exported {basename}")
    plt.clf()
