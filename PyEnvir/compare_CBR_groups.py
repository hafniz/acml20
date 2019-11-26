import os, numpy

datasetPath = "..\\Dataset\\artificial-new\\level0\\rebalanced\\dataset\\R14"
stats = ["filename,min,max,range,mean,median,stddev,var"]

for filename in os.listdir(datasetPath):
    alphas = []
    with open(os.path.join(datasetPath, filename)) as file:
        alphas = [float(row.split(',')[-1]) for row in list(file)[1:]]
    stats.append(f"{filename.split('.')[0]},{numpy.min(alphas)},{numpy.max(alphas)},{numpy.max(alphas) - numpy.min(alphas)},{numpy.mean(alphas)},{numpy.median(alphas)},{numpy.std(alphas)},{numpy.std(alphas) ** 2}")

with open("R14_stats.csv", 'w') as file:
    file.write('\n'.join(stats))
