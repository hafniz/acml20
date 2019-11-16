import os

for datasetGrp in ["LS-DT", "LS-KNN", "LS-NB"]: # 3 files of 4
    os.chdir(f"..\\Dataset\\artificial-new\\MetaLabels\\level1-CV-accy-{datasetGrp}")
    fileAccyVals = [] # 6000 * 10 values
    lvlHeaders = []
    for level in ["two", "four", "six", "eight", "ten", "twelve", "fourteen", "sixteen", "eighteen", "twenty"]:
        lvlHeaders += [level for _ in range(2000)]
    fileAccyVals.append(lvlHeaders)
    for algo in ["DT", "KNN", "NB"]: # 3 columns
        colAccyVals = [] # 6000 values
        for l in range(2, 21, 2): # 10 sections
            level = "{:0>2d}".format(l)
            for var1 in ["A", "B", "V", "G", "D"]:
                for var2 in range(1, 5):
                    filename = f"LS{level}{var1}{var2}-{datasetGrp.split('-')[1]}-{algo}.csv"                 
                    with open(filename) as file:
                        fields = list(file)[0][:-1].split(',')
                        colAccyVals += fields
        fileAccyVals.append(colAccyVals)
    serialized = ""
    for column in list(map(list, zip(*fileAccyVals))):
        serialized += ','.join(column) + '\n'
    with open(f"{datasetGrp}-ACCY-SMRY.csv", 'w') as file:
        file.write(serialized)

for datasetGrp in ["RT"]: # 1 file of 4
    os.chdir(f"..\\Dataset\\artificial-new\\MetaLabels\\level1-CV-accy-{datasetGrp}")
    fileAccyVals = [] # 6000 * 10 values
    lvlHeaders = []
    for level in ["one", "two", "tree", "four", "five", "six", "seven", "eight", "niner", "ten"]:
        lvlHeaders += [level for _ in range(2000)]
    fileAccyVals.append(lvlHeaders)
    for algo in ["DT", "KNN", "NB"]: # 3 columns
        colAccyVals = [] # 6000 values
        for l in range(2, 21, 2): # 10 sections
            level = "{:0>2d}".format(l)
            for v in range(1, 21):
                var = "{:0>2d}".format(v)
                filename = f"RT{level}-{var}-{algo}.csv"              
                with open(filename) as file:
                    fields = list(file)[0][:-1].split(',')
                    colAccyVals += fields
        fileAccyVals.append(colAccyVals)
    serialized = ""
    for column in list(map(list, zip(*fileAccyVals))):
        serialized += ','.join(column) + '\n'
    with open(f"{datasetGrp}-ACCY-SMRY.csv", 'w') as file:
        file.write(serialized)
