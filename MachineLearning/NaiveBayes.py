import math

class NaiveBayes(object):
    def SeparateByClass(dataSet):
        separated = {} # Dictionary<category, List<vector>>
        for i in range(len(dataSet)):
            vector = dataSet[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = [] # Create new category in the dictionary
            separated[vector[-1]].append(vector)
        return separated

    def Average(numbers):
        numbers = [int(number) for number in numbers]
        return sum(numbers) / float(len(numbers))

    def StandardDeviation(numbers):
        numbers = [int(number) for number in numbers]
        return math.sqrt(sum([pow(x - NaiveBayes.Average(numbers), 2) for x in numbers]) / float(len(numbers) - 1))

    def CalculateProbability(x, average, standardDeviation):
        exponent = math.exp(-(math.pow(int(x) - average, 2) / (2 * math.pow(standardDeviation, 2))))
        return (1 / (math.sqrt(2 * math.pi) * standardDeviation)) * exponent

    def Summarize(dataset):
        summaries = [(NaiveBayes.Average(attribute), NaiveBayes.StandardDeviation(attribute)) for attribute in zip(*dataset)] # List<(average, standardDeviation)>
        del summaries[-1]
        return summaries

    def SummarizeByClass(dataSet):
        separated = NaiveBayes.SeparateByClass(dataSet)
        summaries = {} # Dictionary<category, List<(average, standardDeviation)>>
        for category, instances in separated.items():
            summaries[category] = NaiveBayes.Summarize(instances)
        return summaries

    def CalculateClassProbabilities(summaries, vector):
        probabilities = {} # Dictionary<category, int>
        for category, categorySummaries in summaries.items():
            probabilities[category] = 1
            for i in range(len(categorySummaries)):
                average, standardDeviation = categorySummaries[i]
                standardDeviation += 1 # use pseudocount instead
                probabilities[category] *= NaiveBayes.CalculateProbability(vector[i], average, standardDeviation)
        return probabilities

    def Predict(summaries, vector):
        probabilities = NaiveBayes.CalculateClassProbabilities(summaries, vector)
        bestLabel, highestProbablity = None, -1
        for category, probability in probabilities.items():
            if bestLabel is None or probability > highestProbablity:
                highestProbablity = probability
                bestLabel = category
        return bestLabel

    def GetPredictions(summaries, testingSet):
        predictions = []
        for i in range(len(testingSet)):
            result = NaiveBayes.Predict(summaries, testingSet[i])
            predictions.append(result)
        return predictions
