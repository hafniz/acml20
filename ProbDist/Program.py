from sklearn import datasets
from Instance import Instance
from KNNContext import KNNContext
from NaiveBayesContext import NaiveBayesContext
from DecisionTreeContext import DecisionTreeContext
"""
digits = datasets.load_digits()
mlContext = NaiveBayesContext()
mlContext.SetTrainingData(Instance.ReadFromCSV(".\\data\\glass.data", False, True))
mlContext.Train()
print(mlContext.GetProbDist(Instance([1.51937,13.79,2.41,1.19,72.76,0.00,9.77,0.00,0.00])))
"""

from weka.classifiers import IBk
c = IBk(K=1)
c.train("C:\\Users\\CHENH\\Anaconda3\\Lib\\site-packages\\weka\\fixtures\\abalone-train.arff")
predictions = c.predict('query.arff')