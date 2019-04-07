from Instance import Instance
from KNNContext import KNNContext
from NaiveBayesContext import NaiveBayesContext
from DecisionTreeContext import DecisionTreeContext

mlContext = DecisionTreeContext()
mlContext.SetTrainingData(Instance.ReadFromCSV(".\\data\\iris.data"))
mlContext.Train()
print(mlContext.GetProbDist(Instance([6.15, 3.2, 4.9, 1.5])))
