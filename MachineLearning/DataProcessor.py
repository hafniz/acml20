import csv
import random

class DataProcessor(object):
    #region Config
    #attributeCount = 4 # For Iris
    attributeCount = 9 # For Glass
    #endregion

    def LoadData(filename, splitRatio):
        trainingSet = []
        testingSet = []
        with open(filename, "r") as csvFile:
            lines = csv.reader(csvFile)
            dataSet = list(lines)
            for r in range(len(dataSet) - 1):
                #for c in range(DataProcessor.attributeCount): # For Iris
                for c in range(1, 1 + DataProcessor.attributeCount): # For Glass
                    dataSet[r][c] = float(dataSet[r][c])

                # Randomly assign data entry into trainingSet or testingSrt according to splitRatio. 
                if random.random() < splitRatio:
                    trainingSet.append(dataSet[r])
                else:
                    testingSet.append(dataSet[r])
        return (trainingSet, testingSet)
