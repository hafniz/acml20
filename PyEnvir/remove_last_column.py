import os

originPath = "..\\Dataset\\artificial-new\\level0\\rebalanced\\AR14"
destPath = "..\\Dataset\\artificial-new\\level0\\rebalanced\\R14"

for filename in os.listdir(originPath):
    rows = []
    with open(os.path.join(originPath, filename)) as file:
        rows = list(file)
    rows = [','.join(row.split(',')[:-1]) for row in rows]
    with open(os.path.join(destPath, filename), 'w') as file:
        file.write('\n'.join(rows))

