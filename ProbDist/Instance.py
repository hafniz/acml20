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
    def ReadFromCSV(filename, hasHeader = False, hasIndex = False):
        """
        public static List<Instance> ReadFromCsv(string filename, bool hasHeader = false, bool hasIndex = false)
        Reads a CSV file and returns a list of Instance objects.
        filename: Absolute or relative path of the name of the CSV file containing dataset, including filename and file extension.
        hasHeader: True if the first row in the file is header instead of instance data; otherwise, false.
        hasIndex: True if the first column in the file is index instead of feature value; otherwise, false.
        Assumption: Each row in the CSV file represents an entry, where its rightmost value being the label and prior labels being features. 
        """
        instances = []
        with open(filename) as csvFile:
            data = list(csv.reader(csvFile))
            rowRange = range(1, len(data)) if hasHeader else range(len(data))
            for r in rowRange:
                instance = Instance([], None)
                featureCount = len(data[r]) - 1
                columnRange = range(1, featureCount) if hasIndex else range(featureCount)
                for c in columnRange:
                    instance.Features.append(data[r][c]) # TODO: Store the name of feature (in the header) into created Instance object. 
                instance.Label = data[r][-1]
                instances.append(instance)
        return instances
