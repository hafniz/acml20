using System;
using System.Collections.Generic;
using System.Linq;
using static System.Math;

namespace MLCore.Algorithm
{
    public class NaiveBayesContext : AlgorithmContextBase
    {
        public NaiveBayesContext(List<Instance> trainingInstances) : base(trainingInstances) { }
        private readonly Dictionary<string, List<double>> valueStats = new Dictionary<string, List<double>>();
        private readonly Dictionary<string, double> resultProbStats = new Dictionary<string, double>();
        // When featureName == featureValue, at what probability will label == labelValue
        //                       featureName       featureValue           label    prob
        private readonly Dictionary<string, Dictionary<string, Dictionary<string, double>>> factorProbStats = new Dictionary<string, Dictionary<string, Dictionary<string, double>>>();
        private double Interval => Sqrt(TrainingInstances.Count);
        private IEnumerable<string> DistinctLabels => TrainingInstances.Select(i => i.LabelValue ?? throw new NullReferenceException("Unlabeled instance is used as training instance. ")).Distinct();

        // TODO: Refactor this method. 
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
                if (sortedRange[midIndex == 0 ? 0 : midIndex - 1] < value && value < sortedRange[midIndex])
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
            for (int i = 0; i < TrainingInstances.Count; i++)
            {
                foreach (KeyValuePair<string, Feature> kvp in TrainingInstances[i].Features.Where(kvp => kvp.Value.ValueType == ValueType.Discrete))
                {
                    TrainingInstances[i].Features[kvp.Key].ValueDiscretized = kvp.Value.Value;
                }
            }

            foreach (string featureName in TrainingInstances[0].Features.Where(kvp => kvp.Value.ValueType == ValueType.Continuous).Select(kvp => kvp.Key))
            {
                valueStats.Add(featureName, new List<double>());
                foreach (Instance instance in TrainingInstances)
                {
                    valueStats[featureName].Add(instance.Features[featureName].Value);
                }
                valueStats[featureName].Sort();
            }

            //         feature            value  offset
            Dictionary<string, Dictionary<double, int>> valuePositionOffsets = new Dictionary<string, Dictionary<double, int>>();
            TrainingInstances.First().Features.Where(kvp => kvp.Value.ValueType == ValueType.Continuous).ToList().ForEach(kvp => valuePositionOffsets.Add(kvp.Key, new Dictionary<double, int>()));

            foreach (Instance instance in TrainingInstances)
            {
                foreach (KeyValuePair<string, Feature> kvp in instance.Features.Where(kvp => kvp.Value.ValueType == ValueType.Continuous))
                {
                    kvp.Value.ValueDiscretized = PKIDiscretize(kvp.Value.Value, valueStats[kvp.Key], valuePositionOffsets[kvp.Key]);
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

            foreach (string featureName in TrainingInstances.First().Features.Select(kvp => kvp.Key))
            {
                if (!factorProbStats.ContainsKey(featureName))
                {
                    factorProbStats.Add(featureName, new Dictionary<string, Dictionary<string, double>>());
                }

                foreach (string? featureValue in TrainingInstances.Select(i => i.Features[featureName].ValueDiscretized).Distinct())
                {
                    if (featureValue is null)
                    {
                        throw new NullReferenceException("Instance missing discretized feature value. ");
                    }
                    if (!factorProbStats[featureName].ContainsKey(featureValue))
                    {
                        factorProbStats[featureName].Add(featureValue, new Dictionary<string, double>());
                    }
                    foreach (string? label in TrainingInstances.Select(i => i.LabelValue).Distinct())
                    {
                        if (label is null)
                        {
                            throw new NullReferenceException("Unlabeled instance is used as training instance. ");
                        }
                        factorProbStats[featureName][featureValue].Add(label, TrainingInstances.Count(i => i.LabelValue == label && i.Features[featureName].ValueDiscretized == featureValue) / (double)TrainingInstances.Count(i => i.LabelValue == label));
                    }
                }
            }
        }

        public override Dictionary<string, double> GetProbDist(Instance testingInstance)
        {
            foreach (KeyValuePair<string, Feature> kvp in testingInstance.Features)
            {
                testingInstance.Features[kvp.Key].ValueDiscretized = kvp.Value.ValueType == ValueType.Discrete ? kvp.Value.Value : PKIDiscretize(kvp.Value.Value, valueStats[kvp.Key]);
            }

            Dictionary<string, double> probStats = new Dictionary<string, double>();
            foreach (string label in DistinctLabels)
            {
                double priorProb = resultProbStats[label];
                double likelihood = 1;
                foreach (KeyValuePair<string, Feature> kvp in testingInstance.Features)
                {
                    likelihood *= factorProbStats[kvp.Key][kvp.Value.ValueDiscretized ?? throw new NullReferenceException("Feature value not discretized. ")][label];
                }
                double postProbToScale = likelihood * priorProb;
                // postProb = (likelihood * priorProb / evidence) = (postProbToScale / evidence). The denominator is omitted since it is the same for all label values.
                // (Incorrectly) assuming evidence = Features.ForEach(f => evidence *= P(f.Value)), i.e., features are independent of each other
                // will result in value of evidence calculated being lower than actual, thus may making postProb > 1, which is impossible. 
                probStats.Add(label, postProbToScale);
            }
            return OrderedNormalized(probStats);
        }
    }
}
