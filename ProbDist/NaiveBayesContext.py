from Metrics import *

class NaiveBayesContext(object):
    """
    Implementation of Naive Bayes algorithm for multi-class classification.
    Used for features of continuous value. 
    """
    # List<Instance> TrainingInstances
    # Dictionary<TLabel, List<Instance>> SeparatedByLabel
    # Dictionary<TLabel, List<(float mean, float stdDeivation)>> SummarizedByLabel

    def SetTrainingData(self, trainingInstances):
        """
        public void SetTrainingData(List<Instance> trainingInstances)
        Sets the training data for this context. 
        """
        self.TrainingInstances = trainingInstances

    def SeparateByLabel(self):
        """
        private void SeparateByLabel()
        Group trainingInstances by their Label and form a dictionary mapping labels to lists of instances.
        """
        separated = {} # Dictionary<TLabel, List<Instance>>
        for instance in self.TrainingInstances:
            if instance.Label not in separated:
                separated[instance.Label] = []
            separated[instance.Label].append(instance)
        self.SeparatedByLabel = separated
    
    @staticmethod
    def Summarize(instances):
        """
        private static List<(float mean, float stdDeivation)> Summarize(List<Instance> instances)
        Returns: Mean and standard deviation of values of each feature, one feature per tuple. 
        """
        valueCollection = []
        for instance in instances:
            values = []
            for featureValue in instance.Features:
                values.append(featureValue)
            valueCollection.append(values)

        # zip(IEnumberable<IEnumberable> collection) gives the transposition of collection. 
        # e.g., collection = [[v1a, v1b, v1c], 
        #                     [v2a, v2b, v2c], 
        #                     [v3a, v3b, v3c]], 
        # then zip(collection) returns [[v1a, v2a, v3a], 
        #                               [v1b, v2b, v3b], 
        #                               [v1c, v2c, v3c]], 
        # so that the values of the same feature of all the instances are grouped together.
        
        summaries = [(Mean(featureValues), StdDev(featureValues)) for featureValues in zip(*valueCollection)]
        return summaries
    
    def SummarizeByLabel(self):
        """
        private void SummarizeByLabel()
        Returns: A dictionary mapping label to mean and standard deviation of values of each feature of instances of the label. 
        """
        summaries = {}
        for label, instances in self.SeparatedByLabel.items():
            summaries[label] = NaiveBayesContext.Summarize(instances)
        self.SummarizedByLabel = summaries

    def Train(self):
        """
        public void Train()
        Train a model using trainingInstances for testing. 
        """
        self.SeparateByLabel()
        self.SummarizeByLabel()

    def GetProbDist(self, testingInstance):
        """
        public List<(TLabel, float)> GetProbDist(Instance testingInstance)
        Returns: A sorted list of tuples containing probabilities of testingInstance being classified into each of the class in this context.
        """
        # The probability of an instance being classified into a class, for example,  P(label = l1 | feature1 = testingInstance.Feature1, 
        # feature2 = testingInstance.Feature2, ...), is calculated by multiplying together P(feature1 = testingInstance.Feature1 | label = l1), 
        # P(feature2 = testingInstance.Feature2 | label = l2), etc.

        probabilities = {} # Dictionary<TLabel, float>
        for label, labelSummaries in self.SummarizedByLabel.items():
            probabilities[label] = 1
            for i in range(len(labelSummaries)):
                mean, stdDev = labelSummaries[i]
                value = testingInstance.Features[i]
                probabilities[label] *= ProbablityOfBeingIn(value, mean, stdDev)
        return NaiveBayesContext.ToProbDist(probabilities)

    @staticmethod
    def ToProbDist(probabilities):
        """
        private static List<(TLabel, float)> ToProbDist(Dictionary<TLabel, float> probabilities)
        Normalizes values to be sum to 1, in descending order of probabilities. 
        """
        sum = 0.0
        for _, probability in probabilities.items():
            sum += probability
        for label, _ in probabilities.items():
            probabilities[label] /= sum
        return [(label, round(probabilities[label], 3)) for label in sorted(probabilities, key = probabilities.get, reverse = True)]

    def Classify(self, testingInstance):
        """
        public TLabel Classify(Instance testingInstance)
        Returns: The label of the class which testingInstance is most likely to be classified into. 
        """
        return self.GetProbDist(testingInstance)[0][0]
