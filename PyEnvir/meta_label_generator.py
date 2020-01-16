import csv 
import numpy as np 
from scipy import stats 

def paired_t_test(algo1, algo2, name): 
    algo1_acc = algo1[name]
    algo2_acc = algo2[name]

    algo1_acc = np.array(algo1_acc).astype(np.float) 
    algo2_acc = np.array(algo2_acc).astype(np.float)

    t_test = stats.ttest_ind(algo1_acc, algo2_acc) 

    t_test_value = t_test[0]

    # cannot reject null hypothesis 
    if t_test[1] > 0.05:
        t_test_value = 0 

    # if t test result is positive, algo1 is better, if t test result is negative, algo2 is better, if t test result is 0, then draw
    if t_test_value == 0:
        label = 0
    elif t_test_value > 0:
        label = 1
    elif t_test_value < 0:
        label = 2 
    else: 
        label = t_test_value 
        if name not in errorDatasetNames:
            errorDatasetNames.append(name)
            with open(".\\TTestEnvir\\t_test_error.csv", 'a') as f: 
                f.write(name + '\n') 
    return label 

datasetNames = []

with open(".\\TTestEnvir\\knn-accuracy.csv") as f: 
    reader = csv.reader(f)
    knn_acc = {}
    for row in reader:
        knn_acc[row[0]] = row[1:]
    datasetNames = list(knn_acc.keys())

with open(".\\TTestEnvir\\nb-accuracy.csv") as f: 
    reader = csv.reader(f) 
    nb_acc = {}
    for row in reader:
        nb_acc[row[0]] = row[1:]

with open(".\\TTestEnvir\\dt-accuracy.csv") as f: 
    reader = csv.reader(f) 
    dt_acc = {}
    for row in reader:
        dt_acc[row[0]] = row[1:]

t_test_results = {}
for name in datasetNames:
    t_test_results[name] = [None for i in range(3)]

with open(".\\TTestEnvir\\t_test_error.csv", 'w') as f: 
    f.write("datasetName\n")
errorDatasetNames = []

for name in datasetNames: 
    t_test_results[name][0] = paired_t_test(knn_acc, nb_acc, name)
    t_test_results[name][1] = paired_t_test(knn_acc, dt_acc, name)
    t_test_results[name][2] = paired_t_test(nb_acc, dt_acc, name)

with open(".\\TTestEnvir\\paired_t_test_raw_results.csv", 'w') as f: 
    f.write("name,knn-nb,knn-dt,nb-dt\n")
    for name in t_test_results:
        f.write(f"{name},{','.join([str(i) for i in t_test_results[name]])}\n")

# check contradictions and generate labels 
labels_check = [[[-1 for i in range(3)] for j in range(3)] for k in range(3)] 
labels_check[0][0][0] = 0 
labels_check[1][1][0] = 1 
labels_check[1][1][1] = 1 
labels_check[1][1][2] = 1 
labels_check[2][0][1] = 2
labels_check[2][1][1] = 2 
labels_check[2][2][1] = 2 
labels_check[0][2][2] = 3 
labels_check[1][2][2] = 3 
labels_check[2][2][2] = 3 
labels_check[0][1][1] = 4
labels_check[1][0][2] = 5 
labels_check[2][2][0] = 6 

with open(".\\TTestEnvir\\contradictions.csv", 'w') as f: 
    f.write("name,knn-nb,knn-dt,nb-dt\n") 

labels = []
for name in datasetNames: 
    t_test_temp = t_test_results[name] 
    if not t_test_temp[0] in [0,1,2] or not t_test_temp[1] in [0,1,2] or not t_test_temp[2] in [0,1,2]: 
        continue 
    elif labels_check[t_test_temp[0]][t_test_temp[1]][t_test_temp[2]] == -1: 
        with open(".\\TTestEnvir\\contradictions.csv", 'a') as f: 
            f.write(f"{name},{t_test_temp[0]},{t_test_temp[1]},{t_test_temp[2]}\n") 
    else: 
        labels.append(f"{name},{labels_check[t_test_temp[0]][t_test_temp[1]][t_test_temp[2]]}") 

with open(".\\TTestEnvir\\labels.csv", 'w') as f:
    f.write("name,label\n")
    f.write('\n'.join(labels)) 
