import os

datasetPath = "..\\Dataset\\UCI_base_only_ordered\\dataset"
stats = ["filename,instanceCount,featureCount,labelCount,labels"]

for filename in os.listdir(datasetPath):
    rows = []
    with open(os.path.join(datasetPath, filename)) as file:
        rows = list(file)
        rowCount = len(rows)
        columnCount = len(rows[0].split(','))
        labels = []
        for row in rows:
            fields = row[:-1].split(',')
            if len(fields) != columnCount:
                raise Exception("Dataset not rectangular. ")
            label = fields[-1]
            if label not in labels:
                labels.append(label)
        labels.sort()
        stats.append(f"{filename.split('.')[0]},{rowCount},{columnCount - 1},{len(labels)},{','.join(labels)}")
    print(f"Finished calculating {filename}")

with open("TCI_stats.csv", 'w') as file:
        file.write('\n'.join(stats))
