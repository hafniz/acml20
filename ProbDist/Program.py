from Instance import Instance
from KNNContext import KNNContext
from NaiveBayesContext import NaiveBayesContext
from DecisionTreeContext import DecisionTreeContext

mlContext = DecisionTreeContext()
mlContext.SetTrainingData(Instance.ReadFromCSV(".\\data\\glass.data", False, True))
mlContext.Train()
print(mlContext.GetProbDist(Instance([1.51937,13.79,2.41,1.19,72.76,0.00,9.77,0.00,0.00])))
