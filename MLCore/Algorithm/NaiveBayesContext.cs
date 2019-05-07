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

        private string PKIDiscretize(double value, List<double> sortedRange)
        {
            int minIndex = 0;
            int maxIndex = sortedRange.Count - 1;
            double valuePosition = -1;
            while (minIndex <= maxIndex)
            {
                int midIndex = (minIndex + maxIndex) / 2;
                if (sortedRange[0] > value)
                {
                    valuePosition = -0.5;
                    break;
                }
                if (sortedRange[sortedRange.Count - 1] < value)
                {
                    valuePosition = sortedRange.Count - 0.5;
                    break;
                }
                if (sortedRange[minIndex] == value)
                {
                    valuePosition = midIndex;
                    break;
                }
                if (sortedRange[minIndex] < value && sortedRange[minIndex + 1] > value)
                {
                    valuePosition = midIndex + 0.5;
                    break;
                }
                if (sortedRange[minIndex] > value && sortedRange[minIndex - 1] > value)
                {
                    valuePosition = midIndex - 0.5;
                    break;
                }
                if (sortedRange[minIndex] > value)
                {
                    maxIndex = midIndex - 1;
                    continue;
                }
                minIndex = midIndex + 1;
            }
            return $"value discretized into rank {(int)(valuePosition / Interval)}";
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

            foreach (Instance instance in TrainingInstances)
            {
                foreach (KeyValuePair<string, Feature> kvp in instance.Features.Where(kvp => kvp.Value.ValueType == ValueType.Continuous))
                {
                    kvp.Value.Value = PKIDiscretize(kvp.Value.Value, valueStats[kvp.Key]);
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

            double evidence = 1;
            foreach (KeyValuePair<string, Feature> kvp in testingInstance.Features)
            {
                evidence *= TrainingInstances.Count(i => i.Features[kvp.Key].Value == kvp.Value.Value) / (double)TrainingInstances.Count;
            }
            foreach (string label in DistinctLabels)
            {
                double priorProb = resultProbStats[label];
                double likelihood = 1;
                foreach (KeyValuePair<string, Feature> kvp in testingInstance.Features)
                {
                    likelihood *= factorProbStats[kvp.Key][kvp.Value.Value][label];
                }
                probStats.Add(label, likelihood * priorProb / evidence);
            }
            return probStats.OrderByDescending(kvp => kvp.Value).ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }
    }
}
