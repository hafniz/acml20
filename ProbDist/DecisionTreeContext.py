from Metrics import *

class DecisionTreeContext(object):
    """
    Implementation of Naive Bayes algorithm for multi-class classification.
    Used for features of continuous value. 
    Assumptions: GINI impurity; tree depth is max(number of features, number of different labels)
    """
    
    class Node(object):
        """
        Represents a node in a decision tree, consisting of index of feature used to split, threshold value for splitting, and resultant split groups. 
        """

        # int FeatureIndex
        # float Threshold
        # List<Instance> LeftGroup (type may change, see DecisionTreeContext.RecursivelySplit)
        # List<Instance> RightGroup (type may change, see DecisionTreeContext.RecursivelySplit)

        def __init__(self, featureIndex, threshold, leftGroup, rightGroup):
            """
            public Node(int featureIndex, string threshold, List<Instance> leftGroup, List<Instance> rightGroup)
            """
            self.FeatureIndex = featureIndex
            self.Threshold = float(threshold)
            self.LeftGroup = leftGroup
            self.RightGroup = rightGroup

        def __repr__(self):
            """
            public string ToString()
            """
            return "Features[" + str(self.FeatureIndex) + "] < " + str(self.Threshold) + " ? {" + str(len(self.LeftGroup)) + " instances} : {" + str(len(self.RightGroup)) + " instances}"


    # List<Instance> TrainingInstances
    # int MaxDepth
    # Node RootNode

    def SetTrainingData(self, trainingInstances):
        """
        public void SetTrainingData(List<Instance> trainingInstances)
        Sets the training data for this context, and calculates MaxDepth according to self.TrainingInstances. 
        """
        self.TrainingInstances = trainingInstances
        self.MaxDepth = max(len(self.TrainingInstances[0].Features), len(set([instance.Label for instance in self.TrainingInstances])))

    @staticmethod
    def SimpleSplit(instances, featureIndex, threshold):
        """
        private static (List<Instance> leftGroup, List<Instance> rightGroup) Split(List<Instance> instances, int featureIndex, float threshold)
        instances: Source instances to be split into two parts. 
        featureIndex: Index of feature used to split instances. 
        threshold: Threshold value of that feature for splitting instances.
        Returns: A list consisting of two complementary part of instances, namely leftGroup, whose members have value of target feature 
        lower than threshold, and rightGroup, whose members have value of target feature equal to or higher than threshold.
        """
        leftGroup = []
        rightGroup = []
        for instance in instances:
            if float(instance.Features[featureIndex]) < threshold:
                leftGroup.append(instance)
            else: 
                rightGroup.append(instance)
        return (leftGroup, rightGroup)

    @staticmethod
    def GetSplitNode(instances):
        """
        private static Node GetSplitNode(List<Instances> instances)
        instances: Source instances to be split into two parts.
        Returns: The best split criteria in form of Node object, which gives the lowest Gini impurity. 
        """
        labels = list(set([instance.Label for instance in instances]))
        criterialIndex = criterialThreshold = -1
        lowestImpurity = 1
        criterialLeftGroup = criterialRightGroup = None
        for featureIndex in range(len(instances[0].Features)):
            for instance in instances:
                threshold = float(instance.Features[featureIndex])
                splitGroups = DecisionTreeContext.SimpleSplit(instances, featureIndex, threshold)
                (leftGroup, rightGroup) = splitGroups
                giniImpurity = GiniImpurity(splitGroups, labels)
                if giniImpurity < lowestImpurity:
                    criterialIndex = featureIndex
                    criterialThreshold = threshold
                    lowestImpurity = giniImpurity
                    criterialLeftGroup = leftGroup
                    criterialRightGroup = rightGroup
        return DecisionTreeContext.Node(criterialIndex, criterialThreshold, criterialLeftGroup, criterialRightGroup)

    def RecursivelySplit(self, splitNode, currentDepth):
        """
        private void RecursivelySplit(Node splitNode, int currentDepth)
        Recursively split from given splitNode using splitNode until MaxDepth is reached. 
        """
        leftGroup = splitNode.LeftGroup
        rightGroup = splitNode.RightGroup

        if not leftGroup or not rightGroup: # Any of the two is empty List<Instances>
            splitNode.LeftGroup = splitNode.RightGroup = DecisionTreeContext.GetTerminalDist(leftGroup + rightGroup) # Combine two List<Instances> together, allowing duplicate
            return

        if currentDepth >= self.MaxDepth:
            splitNode.LeftGroup = DecisionTreeContext.GetTerminalDist(leftGroup) # Transform group as terminal distribution Dictionary<TLabel, int> when a node is at terminal
            splitNode.RightGroup = DecisionTreeContext.GetTerminalDist(rightGroup)
            return

        splitNode.LeftGroup = DecisionTreeContext.GetSplitNode(leftGroup) # Transform group as splitting Node when a node is not at terminal
        self.RecursivelySplit(splitNode.LeftGroup, currentDepth + 1)

        splitNode.RightGroup = DecisionTreeContext.GetSplitNode(rightGroup)
        self.RecursivelySplit(splitNode.RightGroup, currentDepth + 1)

    def BuildTree(self):
        """
        private void BuildTree()
        Build a decision Tree by recursively splitting self.TrainingInstances. 
        """
        self.RootNode = DecisionTreeContext.GetSplitNode(self.TrainingInstances)
        self.RecursivelySplit(self.RootNode, 1)

    @staticmethod
    def GetTerminalDist(instanceGroup):
        """
        private static Dictionary<TLabel, int> GetTerminalDist(List<Instance> instanceGroup)
        instanceGroup: Instances at the leaf node.
        Returns: A dictionary mapping labels to corresponding occurrences at the leaf node. 
        """
        labels = [instance.Label for instance in instanceGroup]
        labelStats = {}
        for label in labels:
            if label not in labelStats:
                labelStats[label] = 0
            labelStats[label] += 1
        return DecisionTreeContext.ToProbDist(labelStats)

    def PrintTree(self, splitNode, currentDepth = 0):
        if isinstance(splitNode, DecisionTreeContext.Node):
            print(4 * currentDepth * ' ' + "Features[" + str(splitNode.FeatureIndex) + "] < " + str(round(splitNode.Threshold, 3)) + " ?")
            self.PrintTree(splitNode.LeftGroup, currentDepth + 1)
            self.PrintTree(splitNode.RightGroup, currentDepth + 1)
        else:
            print(4 * currentDepth * ' ' + str(splitNode))

    def Train(self):
        """
        public void Train()
        Train a model using trainingInstances for testing. 
        """
        self.BuildTree()
        self.PrintTree(self.RootNode)

    def GetProbDist(self, testingInstance, splitNode = None):
        """
        public List<(TLabel, float)> GetProbDist(Instance testingInstance, Node splitNode)
        Returns: A sorted list of tuples containing probabilities of testingInstance being classified into each of the class in this context.
        """
        if splitNode == None:
            splitNode = self.RootNode
        if testingInstance.Features[splitNode.FeatureIndex] < splitNode.Threshold:
            if isinstance(splitNode.LeftGroup, DecisionTreeContext.Node):
                return self.GetProbDist(testingInstance, splitNode.LeftGroup)
            else:
                return DecisionTreeContext.ToProbDist(splitNode.LeftGroup)
        else:
            if isinstance(splitNode.RightGroup, DecisionTreeContext.Node):
                return self.GetProbDist(testingInstance, splitNode.RightGroup)
            else:
                return DecisionTreeContext.ToProbDist(splitNode.RightGroup)

    @staticmethod
    def ToProbDist(occurrences):
        """
        private static List<(TLabel, float)> ToProbDist(Dictionary<TLabel, int> occurrences)
        Normalizes values to be sum to 1, in descending order of probabilities. 
        """
        sum = 0
        occurrences = dict(occurrences)
        for _, occurrence in occurrences.items():
            sum += occurrence
        for label, _ in occurrences.items():
            occurrences[label] /= sum
        return [(label, round(occurrences[label], 3)) for label in sorted(occurrences, key = occurrences.get, reverse = True)]

    def Classify(self, testingInstance):
        """
        public TLabel Classify(Instance testingInstance)
        Returns: The label of the class which testingInstance is most likely to be classified into. 
        """
        return self.GetProbDist(testingInstance)[0][0]
