import math
import operator

class KNearestNeighbors(object):    
    #region Config
    #attributeCount = 4 # For Iris
    attributeCount = 9 # For Glass
    #endregion

    def CalculateEuclideanDistance(this, other):
        distanceSquared = 0
        #for i in range(KNearestNeighbors.attributeCount): # For Iris
        for i in range(1, 1 + KNearestNeighbors.attributeCount): # For Glass
            distanceSquared += (this[i] - other[i]) ** 2
        return math.sqrt(distanceSquared)

    def GetNeighbors(trainingSet, testInstance, k):
        distances = [] # List<(dataInstance, float)>
        for i in range(len(trainingSet)):
            distance = KNearestNeighbors.CalculateEuclideanDistance(testInstance, trainingSet[i])
            distances.append((trainingSet[i], distance))
        distances.sort(key = operator.itemgetter(1))
        neighbors = [] # List<dataInstance>
        for i in range(k):
            neighbors.append(distances[i][0])
        return neighbors

    def Predict(neighbors):
        categories = {}
        for i in range(len(neighbors)):
            category = neighbors[i][-1]
            if (category in categories):
                categories[category] += 1
            else:
                categories[category] = 1 # Create new category in the dictionary. 
        categoriesSorted = sorted(categories.items(), key = operator.itemgetter(1), reverse = True)
        return categoriesSorted[0][0]
