from Metrics import *

class KNNContext(object):
    """
    Implementation of k-Nearest Neighbors algorithm for multi-class classification
    Used for features of continuous value. 
    Assumptions: k equals to number of all neighbors; uniform vote; Euclidean distance
    """

    # List<Instance> TrainingInstances

    def SetTrainingData(self, trainingInstances):
        """
        public void SetTrainingData(List<Instance> trainingInstances)
        Sets the training data for this context. 
        """
        self.TrainingInstances = trainingInstances

    def Train(self):
        """
        public void Train()
        Does nothing as kNN algorithm does not need training. 
        This method is placed here to conform with the uniform procedure needed at the caller side. 
        """
        pass

    def GetProbDist(self, testingInstance):
        """
        public List<(TLabel, float)> GetProbDist(Instance testingInstance)
        Returns: A sorted list of tuples containing probabilities of testingInstance being classified into each of the class in this context.
        """
        distStats = {} # Dictionary<TLabel, float>
        for instance in self.TrainingInstances:
            if instance.Label not in distStats:
                distStats[instance.Label] = 0
            distStats[instance.Label] += EuclidianDist(instance, testingInstance)
        return KNNContext.ToProbDist(distStats)

    def Classify(self, testingInstance):
        """
        public TLabel Classify(Instance testingInstance)
        Returns: The label of the class which testingInstance is most likely to be classified into. 
        """
        return self.GetProbDist(testingInstance)[0][0]

    @staticmethod
    def ToProbDist(stats):
        """
        private static List<(TLabel, float)> ToProbDist(Dictionary<TLabel, float> stats)
        Normalizes values as inverse of sum of distances, to be sum to 1, and in descending order of probabilities. 
        """
        sum = 0
        for label in stats:
            stats[label] = 1 / stats[label]
            sum += stats[label]
        for label in stats:
            stats[label] /= sum
        return [(label, round(stats[label], 3)) for label in sorted(stats, key = stats.get, reverse = True)]
