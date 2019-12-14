using System;
using System.Collections.Generic;
using System.Linq;
using static System.Math;

namespace MLCore.Algorithm
{
    public class NaiveBayesContext : AlgorithmContextBase
    {
        private readonly int interval;
        private readonly int[] instancesCountInEachInterval;
        private readonly IEnumerable<string> distinctLabels;

        // Smallest and largest values of each interval of each feature
        private readonly Dictionary<string, double[]> intervalBoundaries = new Dictionary<string, double[]>();

        // What proportion of the instances have each of the label values
        //                          label  proportion
        private readonly Dictionary<string, double> resultProbStats = new Dictionary<string, double>();

        // When featureName == featureValue, at what probability will label == labelValue
        //                       featureName       featureValue           label    prob
        private readonly Dictionary<string, Dictionary<string, Dictionary<string, double>>> factorProbStats = new Dictionary<string, Dictionary<string, Dictionary<string, double>>>();

        public NaiveBayesContext(List<Instance> trainingInstances) : base(trainingInstances)
        {
            // 1. Initialize distinct labels
            distinctLabels = TrainingInstances.Select(i => i.LabelValue ?? throw new NullReferenceException("Unlabeled instance is used as training instance. ")).Distinct();

            // 2. Determine number of intervals
            int sqrtInstancesFloor = (int)Sqrt(trainingInstances.Count);
            interval = Sqrt(trainingInstances.Count) == sqrtInstancesFloor ? sqrtInstancesFloor : TrainingInstances.Count < sqrtInstancesFloor * (sqrtInstancesFloor + 1) ? sqrtInstancesFloor : sqrtInstancesFloor + 1;

            // 3. Determine number of instances in each interval
            instancesCountInEachInterval = new int[interval];
            if (TrainingInstances.Count == Pow(interval, 2))
            {
                // e.g. Count = 900, interval = 30 (exact), instancesCountInEachInterval = [30, 30, 30, ..., 30, 30, 30]
                Array.Fill(instancesCountInEachInterval, interval);
            }
            else if (TrainingInstances.Count < Pow(interval, 2))
            {
                // e.g. Count = 890, interval = 30 (underfill), instancesCountInEachInterval = [29, ..., 29, 30, 30, ..., 30, 30]
                int underfilledIntervalCount = (int)Pow(interval, 2) - TrainingInstances.Count;
                Array.Fill(instancesCountInEachInterval, interval - 1, 0, underfilledIntervalCount);
                Array.Fill(instancesCountInEachInterval, interval, underfilledIntervalCount, interval - underfilledIntervalCount);
            }
            else
            {
                // e.g. Count = 910, interval = 30 (overfill), instancesCountInEachInterval = [30, 30, ..., 30, 30, 31, ..., 31]
                int overfilledIntervalCount = trainingInstances.Count - (int)Pow(interval, 2);
                Array.Fill(instancesCountInEachInterval, interval, 0, interval - overfilledIntervalCount);
                Array.Fill(instancesCountInEachInterval, interval + 1, interval - overfilledIntervalCount, overfilledIntervalCount);
            }
        }

        public override void Train()
        {
            // 1. Discretize training instances and fill in intervalBoundaries
            for (int i = 0; i < TrainingInstances.Count; i++)
            {
                foreach (Feature feature in TrainingInstances[i].Features.Where(f => f.ValueType == ValueType.Discrete))
                {
                    TrainingInstances[i][feature.Name].ValueDiscretized = feature.Value;
                }
            }

            foreach (string featureName in TrainingInstances.First().Features.Where(f => f.ValueType == ValueType.Continuous).Select(f => f.Name))
            {
                List<(Instance instance, double featureValue)> featureValues = new List<(Instance, double)>();
                foreach (Instance instance in TrainingInstances)
                {
                    featureValues.Add((instance, instance[featureName].Value));
                }
                featureValues.Sort((tuple1, tuple2) => tuple1.featureValue.CompareTo(tuple2.featureValue));
                int finishDiscretizedCount = 0;
                intervalBoundaries[featureName] = new double[interval];
                for (int i = 0; i < interval; i++)
                {
                    intervalBoundaries[featureName][i] = featureValues[finishDiscretizedCount].instance[featureName].Value;
                    for (int j = 0; j < instancesCountInEachInterval[i]; j++)
                    {
                        featureValues[finishDiscretizedCount].instance[featureName].ValueDiscretized = $"interval{i}";
                        finishDiscretizedCount++;
                    }
                }
            }

            // 2. Fill in resultProbStats
            foreach (string label in distinctLabels)
            {
                resultProbStats.Add(label, TrainingInstances.Count(i => i.LabelValue == label) / (double)TrainingInstances.Count);
            }

            // 3. Fill in factorProbStats
            foreach (string featureName in TrainingInstances.First().Features.Select(f => f.Name))
            {
                if (!factorProbStats.ContainsKey(featureName))
                {
                    factorProbStats.Add(featureName, new Dictionary<string, Dictionary<string, double>>());
                }

                foreach (string? featureValue in TrainingInstances.Select(i => i[featureName].ValueDiscretized).Distinct())
                {
                    if (!factorProbStats[featureName].ContainsKey(featureValue ?? throw new NullReferenceException("Instance missing discretized feature value. ")))
                    {
                        factorProbStats[featureName].Add(featureValue, new Dictionary<string, double>());
                    }
                    foreach (string? label in TrainingInstances.Select(i => i.LabelValue).Distinct())
                    {
                        factorProbStats[featureName][featureValue].Add(label ?? throw new NullReferenceException("Unlabeled instance is used as training instance. "), TrainingInstances.Count(i => i.LabelValue == label && i[featureName].ValueDiscretized == featureValue) / (double)TrainingInstances.Count(i => i.LabelValue == label));
                    }
                }
            }
        }

        public override Dictionary<string, double> GetProbDist(Instance testingInstance)
        {
            // 1. Discretize testing instance
            foreach (Feature feature in testingInstance.Features.Where(f => f.ValueType == ValueType.Discrete))
            {
                testingInstance[feature.Name].ValueDiscretized = feature.Value;
            }

            foreach (Feature feature in testingInstance.Features.Where(f => f.ValueType == ValueType.Continuous))
            {
                double continuousValue = feature.Value;
                if (continuousValue < intervalBoundaries[feature.Name][0])
                {
                    testingInstance[feature.Name].ValueDiscretized = "interval0";
                }
                else if (continuousValue >= intervalBoundaries[feature.Name][^1])
                {
                    testingInstance[feature.Name].ValueDiscretized = $"interval{interval - 1}";
                }
                else
                {
                    for (int i = 0; i < interval; i++)
                    {
                        if (continuousValue < intervalBoundaries[feature.Name][i + 1] && continuousValue >= intervalBoundaries[feature.Name][i])
                        {
                            testingInstance[feature.Name].ValueDiscretized = $"interval{i}";
                            break;
                        }
                    }
                    if (testingInstance[feature.Name].ValueDiscretized is null)
                    {
                        throw new Exception($"Failed to discretize {feature.Name} ({continuousValue})");
                    }
                }
            }

            // 2. Calculate probabilities
            Dictionary<string, double> probStats = new Dictionary<string, double>();
            foreach (string label in distinctLabels)
            {
                double priorProb = resultProbStats[label];
                double likelihood = 1;
                foreach (Feature feature in testingInstance.Features)
                {
                    likelihood *= factorProbStats[feature.Name][feature.ValueDiscretized ?? throw new NullReferenceException("Feature value not discretized. ")][label];
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
