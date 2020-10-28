import os
from scipy import stats 

def paired_t_test(accuracies1, accuracies2): 
    t_test = stats.ttest_ind(accuracies1, accuracies2) 
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
    return label, t_test[0], t_test[1]

values = []
with open(f"{os.path.expanduser('~\\Desktop')}\\UCI analysis\\accy\\6.csv") as file:
    values = [[float(value) for value in row.split(',')] for row in list(file)]

print("left(1)/draw(0)/right(2),t-statistic,p-value")
for i in range(1, 12):
    label, tStats, pValue = paired_t_test(values[0], values[i])
    print(f"Set1 vs Set{i + 1},{label},{tStats},{pValue}")
