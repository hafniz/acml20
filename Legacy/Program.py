from DataProcessor import *
from KNearestNeighbors import *
from NaiveBayes import *
from DecisionTree import *
from TestMetrics import *

#region Config
#filename = ".\data\Iris\iris.data"
filename = ".\data\Glass\glass.data"
splitRatio = 0.67
kMinimum = 1
kMaximum = 10
#endregion

print("Running... ")
(trainingSet, testingSet) = DataProcessor.LoadData(filename, splitRatio)

def RunKNearsetNeighbors():
    for k in range(kMinimum, kMaximum + 1):
        predictions = [] # List<Dictionary<string, int>>
        for i in range(len(testingSet)):
            neighbors = KNearestNeighbors.GetNeighbors(trainingSet, testingSet[i], k)
            prediction = KNearestNeighbors.Predict(neighbors)
            predictions.append(prediction)
            #print("Predicted: ", prediction, "Actual: ", testingSet[i][-1])
        accuracy = TestMetrics.GetPercentageAccuracy(testingSet, predictions)
        print("Accuracy:", str(round(accuracy, 2)) + "%", "(k = " + str(k) + ")")

def RunNaiveBayes():
    summaries = NaiveBayes.SummarizeByClass(trainingSet)
    predictions = NaiveBayes.GetPredictions(summaries, testingSet)
    accuracy = TestMetrics.GetPercentageAccuracy(testingSet, predictions)
    print("Accuracy:", str(round(accuracy, 2)) + "%")

RunNaiveBayes()
