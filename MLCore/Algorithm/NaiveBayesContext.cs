using System.Collections.Generic;
using System.Linq;
using static System.Math;

namespace MLCore.Algorithm
{
    public class NaiveBayesContext : AlgorithmContextBase
    {
        public NaiveBayesContext(List<Instance> trainingInstances) : base(trainingInstances) { }
        private Dictionary<string, List<double>> valueStats = new Dictionary<string, List<double>>();
        private Dictionary<string, double> resultProbStats = new Dictionary<string, double>();
        private Dictionary<string, Dictionary<string, Dictionary<string, double>>> factorProbStats = new Dictionary<string, Dictionary<string, Dictionary<string, double>>>();
        private double Interval => Sqrt(TrainingInstances.Count);
#pragma warning disable CS8619 // Nullability of reference types in value doesn't match target type.
        // This is suppressed because trainingInstance definitely has a non-null LabelValue. 
        private IEnumerable<string> DistinctLabels => TrainingInstances.Select(i => i.LabelValue).Distinct();
#pragma warning restore CS8619 // Nullability of reference types in value doesn't match target type.

        private string PKIDiscretize(double value, List<double> sortedRange, Dictionary<double, int>? valuePositionOffset = null)
        {
            if (value < sortedRange[0])
            {
                return $"interval0";
            }
            if (sortedRange[^1] < value)
            {
                return $"interval{(int)Interval}";
            }

            int minIndex = 0;
            int maxIndex = sortedRange.Count - 1;
            double valuePosition = -1;
            while (minIndex <= maxIndex)
            {
                int midIndex = (minIndex + maxIndex) / 2;
                if (sortedRange[midIndex] == value)
                {
                    valuePosition = midIndex;

                    // This is to resolve the problem of having multiple duplicate values for a certain feature in the training instances that 
                    // their valuePositions may cross the border(s) of intervals. 
                    if (valuePositionOffset is null)
                    {
                        // In testing phase, assign interval based on the median valuePosition of the value in the training instances. 

                        double minPosition = valuePosition;
                        while (minPosition != 0 && sortedRange[(int)minPosition - 1] == value)
                        {
                            minPosition--;
                        }
                        double maxPosition = valuePosition;
                        while (maxPosition != sortedRange.Count - 1 && sortedRange[(int)maxPosition + 1] == value)
                        {
                            maxPosition++;
                        }
                        valuePosition = (minPosition + maxPosition) / 2;
                    }
                    else
                    {
                        // In training phase, assign interval base on the order of occurrence in sortedRange. E.g., the first occurrence of 
                        // a certain value will get its valuePosition of the first occurrence in sortedRange, say i. The second occurrence
                        // of the same value will then get i + 1.

                        while (valuePosition != 0 && sortedRange[(int)valuePosition - 1] == value)
                        {
                            valuePosition--;
                        }
                        if (!(valuePositionOffset.ContainsKey(value)))
                        {
                            valuePositionOffset.Add(value, -1);
                        }
                        valuePositionOffset[value]++;
                        valuePosition += valuePositionOffset[value];
                    }
                    break;
                }
                if (sortedRange[midIndex] < value && value < sortedRange[midIndex + 1])
                {
                    return $"interval{(int)((midIndex + 0.5) / Interval)}";
                }
                if (sortedRange[midIndex - 1] < value && value < sortedRange[midIndex + 1])
                {
                    return $"interval{(int)((midIndex - 0.5) / Interval)}";
                }
                if (value < sortedRange[midIndex])
                {
                    maxIndex = midIndex - 1;
                    continue;
                }
                minIndex = midIndex + 1;
            }
            return $"interval{(int)(valuePosition / Interval)}";
        }
        private void DiscretizeTrainingInstances()
        {
            foreach (string featureName in TrainingInstances[0].Features.Where(kvp => kvp.Value.ValueType == ValueType.Continuous).Select(kvp => kvp.Key))
            {
                valueStats.Add(featureName, new List<double>());
                foreach (Instance instance in TrainingInstances)
                {
                    valueStats[featureName].Add(instance.Features[featureName].Value);
                }
                valueStats[featureName].Sort();
            }

            Dictionary<string, Dictionary<double, int>> valuePositionOffsets = new Dictionary<string, Dictionary<double, int>>();
            TrainingInstances[0].Features.Where(kvp => kvp.Value.ValueType == ValueType.Continuous).ToList().ForEach(kvp => valuePositionOffsets.Add(kvp.Key, new Dictionary<double, int>()));

            foreach (Instance instance in TrainingInstances)
            {
                foreach (KeyValuePair<string, Feature> kvp in instance.Features.Where(kvp => kvp.Value.ValueType == ValueType.Continuous))
                {
#warning Calling PKIDiscretize in this way will override original continuous input feature values irreversibly. To be fixed later. 
                    kvp.Value.Value = PKIDiscretize(kvp.Value.Value, valueStats[kvp.Key], valuePositionOffsets[kvp.Key]);
                    kvp.Value.ValueType = ValueType.Discrete;
                }
            }
        }
        public override void Train()
        {
            DiscretizeTrainingInstances();

            foreach (string label in DistinctLabels)
            {
                resultProbStats.Add(label, TrainingInstances.Count(i => i.LabelValue == label) / (double)TrainingInstances.Count);
            }

            foreach (string featureName in TrainingInstances[0].Features.Select(kvp => kvp.Key))
            {
                if (!factorProbStats.ContainsKey(featureName))
                {
                    factorProbStats.Add(featureName, new Dictionary<string, Dictionary<string, double>>());
                }
                foreach (string featureValue in TrainingInstances.Select(i => i.Features[featureName].Value).Distinct())
                {
                    if (!factorProbStats[featureName].ContainsKey(featureValue))
                    {
                        factorProbStats[featureName].Add(featureValue, new Dictionary<string, double>());
                    }
                    foreach (string label in TrainingInstances.Select(i => i.LabelValue).Distinct())
                    {
                        factorProbStats[featureName][featureValue].Add(label, TrainingInstances.Count(i => i.LabelValue == label && i.Features[featureName].Value == featureValue) / (double)TrainingInstances.Count(i => i.LabelValue == label));
                    }
                }
            }
        }
        public override Dictionary<string, double> GetProbDist(Instance testingInstance)
        {
            Dictionary<string, double> probStats = new Dictionary<string, double>();
            foreach (KeyValuePair<string, Feature> kvp in testingInstance.Features.Where(kvp => kvp.Value.ValueType == ValueType.Continuous))
            {
                kvp.Value.Value = PKIDiscretize(kvp.Value.Value, valueStats[kvp.Key]);
            }

            foreach (string label in DistinctLabels)
            {
                double priorProb = resultProbStats[label];
                double likelihood = 1;
                foreach (KeyValuePair<string, Feature> kvp in testingInstance.Features)
                {
                    likelihood *= factorProbStats[kvp.Key][kvp.Value.Value][label];
                }
                double postProbToScale = likelihood * priorProb;
                // postProb = likelihood * priorProb / evidence = postProbToScale / evidence. The denominator is omitted since it is the same for all label values.
                // (Incorrectly) assuming evidence = Features.ForEach(f => evidence *= P(f.Value)), i.e., each feature value being independent
                // will result in value of evidence calculated being lower than actual, thus may making posstProb > 1, which is impossible. 
                probStats.Add(label, postProbToScale);
            }
            return OrderedNormalized(probStats);
        }
    }
}
