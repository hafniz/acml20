import csv

class Instance(object):
    """
    Represents an instance/data entry having both feature vector and output label (pre-constructed for training instances, assigned by algorithm for testing instances). 
    """

    # TLabel label;
    # List<string> features;

    def __init__(self, features = [], label = None):
        """
        public Instance(List<string> features, TLabel label)
        """
        self.Features = features
        self.Label = label

    def __repr__(self):
        """
        public string ToString()
        """
        return str(self.Features) + ", " + str(self.Label)

    @staticmethod
    def ReadFromCSV(filename):
        """
        public static List<Instance> ReadFromCsv(string filename)
        Reads a CSV file and returns a list of Instance objects.
        Assumption: Each row in the CSV file represents an entry, where its rightmost value being the label and prior labels being features. 
        """
        instances = []
        with open(filename) as csvFile:
            data = list(csv.reader(csvFile))
            for r in range(len(data)):
                instance = Instance([], None)
                featureCount = len(data[r]) - 1
                for c in range(featureCount):
                    instance.Features.append(data[r][c])
                instance.Label = data[r][-1]
                instances.append(instance)
        return instances
