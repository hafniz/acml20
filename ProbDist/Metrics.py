import math

def EuclidianDist(instance1, instance2):
    """
    public static float EuclidianDist(Instance instance1, Instance instance2)
    Assumptions: All members of instance1.Features and instance2.Features are of numerical type. 
    """
    return sum([(float(instance1.Features[i]) - float(instance2.Features[i])) ** 2 for i in range(len(instance1.Features))]) ** 0.5

def Mean(numbers):
    """
    public static float Mean(List<string> numbers)
    """
    return sum([float(n) for n in numbers]) / float(len(numbers))

def StdDev(numbers):
    """
    public static float StdDev(List<string> numbers)
    Returns: The standard deviation of numbers.
    """
    average = Mean(numbers)
    variance = sum([(float(x) - average) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def ProbablityOfBeingIn(value, mean, stdDev):
    """
    public static float ProbablityOfBeingIn(float value, float mean, float stdDev)
    Returns: The probability of value falling in the category in which values are distributed according to N(mean, stdDev^2). 
    Used in Naive Bayes to get, for example, P(feature1 = v1 | label = l1)
    """
    exponent = math.exp(-(((value - mean) ** 2)/(2 * (stdDev ** 2))))
    return (1 / (((2 * math.pi) ** 0.5) * stdDev)) * exponent

def GiniImpurity(instanceGroups, instanceLabels):
    """
    public static float GiniImpurity(List<List<Instance>> instanceGroups, List<TLabel> instanceLabels)
    Returns: The Gini impurity of the split dataset. 
    """
    totalInstances = float(sum([len(instanceGroup) for instanceGroup in instanceGroups]))
    giniImpurity = 0.0
    for instanceGroup in instanceGroups:
        groupSize = float(len(instanceGroup))
        if groupSize == 0:
            continue
        proportionSum = 0.0
        for label in instanceLabels:
            proportion = [instance.Label for instance in instanceGroup].count(label) / groupSize
            proportionSum += proportion ** 2
        giniImpurity += (1.0 - proportionSum) * (groupSize / totalInstances)
    return giniImpurity
