import os

datasetPath = "..\\Dataset\\artificial-new\\level0\\rebalanced\\R14"
filenames = [os.path.join(datasetPath, filename) for filename in os.listdir(datasetPath)]

stats = ["filename,#pos,%pos,#neg,%neg,CBR,count,NIR"]
for filename in filenames:
    rows = []
    count = -1
    with open(filename) as file:
        rows = list(file)
        count = len(rows)
    labels = [float(row.split(',')[2]) for row in rows]
    pos = labels.count(1.0)
    neg = labels.count(0.0)
    cbr = pos / neg if pos < neg else neg / pos
    stats.append(f"{os.path.basename(filename).split('.')[0]},{pos},{pos / count},{neg},{neg / count},{cbr},{count},{count / 2500}")

with open("class_balance_r14.csv", 'w') as file:
    file.write('\n'.join(stats))
