from sklearn import datasets
from Instance import Instance
from KNNContext import KNNContext
from NaiveBayesContext import NaiveBayesContext
from DecisionTreeContext import DecisionTreeContext

datasets.load_iris()

digits = datasets.load_digits()
mlContext = NaiveBayesContext()
mlContext.SetTrainingData(Instance.ReadFromCSV(".\\data\\glass.data", False, True))
mlContext.Train()
print(mlContext.GetProbDist(Instance([1.51937,13.79,2.41,1.19,72.76,0.00,9.77,0.00,0.00])))