import time 
import numpy as np
import pandas as pd
from os import listdir
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.tree import DecisionTreeClassifier

def filter_attribute(X):
    continous_cols = []
    nominal_cols = []
    for i in range(X.shape[1]):
        col = X[:, i]
        
        not_cont = True
        for ele in col:
            if ele == 0 or ele == 1:
                pass
            else:
                not_cont = False

        if not_cont:
            nominal_cols.append(col)
        else:
            continous_cols.append(col)
            
    return continous_cols, nominal_cols

# Information theoretic

def entropy(a):
    probs = [np.mean(a == c) for c in set(a)]
    return np.sum(-p * np.log2(p) for p in probs if p > 0)

def joint_entropy_2val(a, b):
    probs = []
    for c1 in set(a):
        for c2 in set(b):
            probs.append(np.mean(np.logical_and(a == c1, b == c2)))

    return np.sum(-p * np.log2(p) for p in probs if p > 0)

def infoMetas(dataset):
    X = dataset[:, :-1]
    y = dataset[:, -1]
    
    cont, nom = filter_attribute(X)
    
    #Convert continous columns into frequency bins
    for attr in cont:
        attr_vals = pd.cut(attr, 10, labels=list(range(10)), retbins=True)[0]
        nom.append(attr_vals)
    
    class_ent = entropy(y)
    
    attr_entr = np.zeros(len(nom))
    attr_entr_norm = np.zeros(attr_entr.shape)
    mut_info = np.zeros(attr_entr.shape)
    joint_entr = np.zeros(attr_entr.shape)
    
    for i in range(len(nom)):
        attr  = nom[i]
        attr_entr_cur = entropy(attr)
        attr_entr[i] = attr_entr_cur
        attr_entr_norm[i] = attr_entr_cur / np.log2(len(attr))
        
        joint_entr_cur = joint_entropy_2val(attr, y)
        joint_entr[i] = joint_entr_cur
        
        mut_info[i] = class_ent + attr_entr_cur - joint_entr_cur
    
    '''ClassEnt, AttrEnt[Min, Mean, Max], JointEnt, MutInfo[Min, Mean, Max], EquiAttr, NoiseRatio'''
    metas_list = [class_ent, attr_entr_norm.min(), attr_entr_norm.mean(), attr_entr_norm.max(), joint_entr.mean(), 
                  mut_info.min(), mut_info.mean(), mut_info.max(), class_ent / mut_info.mean(), 
                  np.abs(attr_entr.mean() - mut_info.mean()) / mut_info.mean()]
    return metas_list

# Statistical

def statMetas(dataset):
    X = dataset[:, :-1]
    y = dataset[:, -1]
    
    #Include all columns as continous attributes
    cont = X
        
    std_x = np.zeros(len(cont))
    skew_x = np.zeros(std_x.shape)
    kurtosis_x = np.zeros(std_x.shape)

    for i in range(len(cont)):
        attr = cont[i]
        std_x[i] = attr.std()
        skew_x[i] = skew(attr)
        kurtosis_x[i] = kurtosis(attr)
    
    '''StandardDev[Min, Mean, Max], Skewness[Min, Mean, Max], Kurtosis[Min, Mean, Max]'''
    metas_list = [std_x.min(), std_x.mean(), std_x.max(), skew_x.min(), skew_x.mean(), skew_x.max(), 
                  kurtosis_x.min(), kurtosis_x.mean(), kurtosis_x.max()]
    return metas_list

# Decision Tree

def decisionTreeMetas(dataset):
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    
    clf = DecisionTreeClassifier()
    clf.fit(X, Y)

    model = clf.tree_
    n_nodes = model.node_count
    children_left = model.children_left
    children_right = model.children_right
    feature = model.feature
    threshold = model.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    n_leaves = 0
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
            n_leaves += 1

    #Tree Width
    num_left = 0
    cur = 0
    while cur != -1:
        cur = model.children_left[cur]
        num_left += 1

    num_right = 0
    cur = 0
    while cur != -1:
        cur = model.children_right[cur]
        num_right += 1

    n_width = num_left + num_right - 1

    #Branches
    branch_length = []
    for i in range(n_nodes):
        if is_leaves[i]:
            branch_length.append(node_depth[i])
    branch_length = np.array(branch_length)

    #Nodes per level
    nodes_level = np.zeros(shape=model.max_depth)
    for i in node_depth[1:]:
        nodes_level[i-1] += 1

    #Attribute occurence
    feature_occr = np.zeros(shape=clf.n_features_)
    for i in feature:
        if not i == -2:
            feature_occr[i] += 1

    #Output
    '''treewidth, treeheight, NoNode, NoLeave, maxLevel, meanLevel, devLevel, 
       ShortBranch, meanBranch, devBranch, maxAtt, minAtt, 
       meanAtt, devAtt'''
    metas_list = [n_width, model.max_depth, n_nodes, n_leaves, max(nodes_level), nodes_level.mean(), nodes_level.std(),
                  min(branch_length), branch_length.mean(), branch_length.std(), max(feature_occr), min(feature_occr), 
                  feature_occr.mean(), feature_occr.std()]
    return metas_list

# Calculating and saving

dataset_folder = input('dataset folder: ')
results_folder = input('results folder: ')
timing_filename = input('timing filename: ')
header_filename = input('header filename: ')

count = 0
timing_stats = []
total_time = 0.0
datasets = listdir(dataset_folder)

for filename in datasets:

    dataset = pd.read_csv(dataset_folder + '\\' + filename, header = None).values
    
    try:
        start = time.time()
        temp = infoMetas(dataset) + statMetas(dataset) + decisionTreeMetas(dataset)
        duration = time.time() - start
        total_time += duration
        count += 1
        print(f'{count}\t{filename}\t{duration}\t{total_time}')
        timing_stats.append(f'{filename},{duration},{total_time}\n')
    except ValueError:
        print('Dataset without continous attributes found: ', filename)
    else:
        has_nan = np.any(np.isnan(temp))
        has_inf = np.any(np.isinf(temp))
        if has_nan and has_inf:
            print('Dataset has both nan and inf: ', filename)
            break
        elif has_nan:
            print('Dataset has nan: ', filename)
            break
        elif has_inf:
            print('Dataset has inf: ', filename)
            break
    
    temp = np.array(temp)
    temp.dump(results_folder + '\\' + filename + '.npy')

with open(timing_filename, 'w') as f: 
    f.writelines(timing_stats) 

names_list = 'ClassEnt, AttrEntMin, AttrEntMean, AttrEntMax, JointEnt, MutInfoMin, MutInfoMean, MutInfoMax, EquiAttr,' + \
             'NoiseRatio, StandardDevMin, StandardDevMean, StandardDevMax, SkewnessMin, SkewnessMean, SkewnessMax,' + \
             'KurtosisMin, KurtosisMean, KurtosisMax, treewidth, treeheight, NoNode, NoLeave, maxLevel, meanLevel, devLevel,' + \
             'ShortBranch, meanBranch, devBranch, maxAtt, minAtt, meanAtt, devAtt'
names_list = np.array([x.strip() for x in names_list.split(',')])

names_list.dump(header_filename)
