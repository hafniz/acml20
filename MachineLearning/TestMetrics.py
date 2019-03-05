class TestMetrics(object):
    def GetPercentageAccuracy(testingSet, predictions):
        correct = 0
        for i in range(len(testingSet)):
            if (testingSet[i][-1] == predictions[i]):
                correct += 1
        return (correct / len(testingSet)) * 100.0
