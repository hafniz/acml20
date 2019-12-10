import os

results = ["filename,bin0,bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9"]
finishedCount = 0

for filename in os.listdir("..\\Dataset\\artificial-R\\dataset2824 with alpha"):
    rows = []
    with open(os.path.join("..\\Dataset\\artificial-R\\dataset2824 with alpha", filename)) as file:
        rows = list(file)[1:]
    instanceCount = len(rows)
    binFreq = [0 for _ in range(10)]
    for row in rows:
        alpha = float(row.split(',')[-1])
        binFreq[9 if alpha == 1 else int(alpha * 10)] += 1
    binFreq = [i / instanceCount for i in binFreq]
    results.append(f"{filename.split('.')[0]},{','.join([str(d) for d in binFreq])}")
    finishedCount += 1
    print(finishedCount)

with open("..\\ar2824_binFreq.csv", 'w') as file:
    file.write('\n'.join(results))
