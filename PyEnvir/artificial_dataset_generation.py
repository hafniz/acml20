debug = False 

import os, csv, random 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 

home = os.getcwd() 
dataset_path = os.path.join(home, "datasets") 
if not os.path.exists(dataset_path): 
	os.mkdir(dataset_path) 
os.chdir(dataset_path)

def get_dist(a, b): 
	#for instance i, its position is (i//25*1/24, i%25*1/24) 
	x_a = (a//25) * (1/24) 
	y_a = (a%25) * (1/24) 
	x_b = (b//25) * (1/24) 
	y_b = (b%25) * (1/24) 
	return pow(x_a-x_b, 2) + pow(y_a-y_b, 2)  

def get_min(l): 
	if len(l) <2: 
		return l 
	else: 
		less = [] 
		more = [] 
		pivot = l[0] 
		for i in range(1, len(l)): 
			if l[i][1] <= pivot[1]: 
				less.append(l[i]) 
			else: 
				more.append(l[i]) 
	return get_min(less) + [pivot] + get_min(more) 

#define the complexity k, range from 1 to 9
for k in range(1, 10): 
	print("k", k)
	#each complexity level would have 10 datasets 
	for n_dataset in range(10): 
		print("n_dataset", n_dataset)
		#define the dataset
		dataset_x = [[(i//25)*(1/24), (i%25)*(1/24)] for i in range(25*25)] 
		dataset_y = [None for i in range(25*25)]

		#divide the grid into 2^k subsections 
		sections = [[] for i in range(pow(2,k))] 
		num = [i for i in range(25*25)] 
		random.shuffle(num) 
		j = 0
		for i in range(25*25): 
			sections[j].append(num[i])
			if j == (pow(2,k)-1): 
				j = 0 
			else: 
				j += 1 

		#generate the centroid points 
		centroids = [] 
		for section in sections: 
			#for instance i, its position is (i//25*1/24, i%25*1/24) 
			dis = [] 
			for instance in section: 
				dis_temp = 0 
				for point in section: 
					if instance != point: 
						dis_temp += get_dist(point, instance) 
				dis.append([instance, dis_temp]) 
			min_dis = get_min(dis) 
			centroids.append(min_dis[0][0])  

		if debug: 
			print(centroids) 

		#give each centroid point a random label 
		for centroid in centroids: 
			dataset_y[centroid] = random.choice([0,1]) 

		#use 3 different algorithms to build the model and assign labels to instances based on the models 
		#generate the training set and testing set 
		test_x = [] 
		test_y = [] 
		train_x = [] 
		train_y = [] 
		for centroid in centroids: 
			train_x.append(dataset_x[centroid]) 
			train_y.append(dataset_y[centroid]) 
		for i in range(25*25): 
			if i not in centroids: 
				test_x.append(dataset_x[i]) 

		#knn 
		classifier = KNeighborsClassifier(n_neighbors=int(pow(pow(2,k), 0.5))) 
		classifier.fit(train_x, train_y) 
		test_y = classifier.predict(test_x) 

		dataset_name = str((k-1)*10+n_dataset) + "_knn.csv" 
		with open(dataset_name, "w", newline="") as f: 
			writer = csv.writer(f) 
			temp = [] 
			for i in range(len(train_x)): 
				row = train_x[i][:] 
				row.append(train_y[i]) 
				temp.append(row) 
			for i in range(len(test_x)): 
				row = test_x[i][:] 
				row.append(test_y[i]) 
				temp.append(row)
			writer.writerows(temp) 

		#decision tree 
		classifier = DecisionTreeClassifier()
		classifier.fit(train_x, train_y) 
		test_y = classifier.predict(test_x) 

		dataset_name = str((k-1)*10+n_dataset) + "_dt.csv" 
		with open(dataset_name, "w", newline="") as f: 
			writer = csv.writer(f) 
			temp = [] 
			for i in range(len(train_x)): 
				row = train_x[i][:] 
				row.append(train_y[i]) 
				temp.append(row) 
			for i in range(len(test_x)): 
				row = test_x[i][:] 
				row.append(test_y[i]) 
				temp.append(row) 
			writer.writerows(temp) 

		#naive bayes 
		classifier = GaussianNB()
		classifier.fit(train_x, train_y) 
		test_y = classifier.predict(test_x) 

		dataset_name = str((k-1)*10+n_dataset) + "_nb.csv" 
		with open(dataset_name, "w", newline="") as f: 
			writer = csv.writer(f) 
			temp = [] 
			for i in range(len(train_x)): 
				row = train_x[i][:] 
				row.append(train_y[i]) 
				temp.append(row) 
			for i in range(len(test_x)): 
				row = test_x[i][:] 
				row.append(test_y[i]) 
				temp.append(row) 
			writer.writerows(temp) 
