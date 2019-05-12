using System;
using System.Collections.Generic;
using System.Linq;
using static System.Math;

namespace MLCore.Algorithm
{
    public class DecisionTreeContext : AlgorithmContextBase
    {
        class Node
        {
            public string SplitFeatureName { get; private set; }
            public double? SplitThreshold { get; private set; }
            public bool IsLeafNode { get; set; }
            public List<Instance> InstancesIn { get; set; }
            public Dictionary<string, double> LeafProbDist { get; set; } = new Dictionary<string, double>();
            public Dictionary<string, Node> SubNodes { get; set; } = new Dictionary<string, Node>();

            public Node(List<Instance> instancesIn, string splitFeatureName, double? splitThreshold)
            {
                InstancesIn = instancesIn;
                SplitFeatureName = splitFeatureName;
                SplitThreshold = splitThreshold;
            }
            public Node NavigateDown(Instance instance)
            {
                if (SubNodes is null)
                {
                    throw new NullReferenceException("SubNodes is null. ");
                }
                if (instance.Features[SplitFeatureName].ValueType == ValueType.Discrete)
                {
                    return SubNodes[instance.Features[SplitFeatureName].Value];
                }
                if (instance.Features[SplitFeatureName].Value <= SplitThreshold)
                {
                    return SubNodes["less than or equal to threshold"];
                }
                return SubNodes["greater than threshold"];
            }
            public void CalcLeafProbDist()
            {
                foreach (string label in InstancesIn.Select(i => i.LabelValue).Distinct())
                {
                    LeafProbDist.Add(label, InstancesIn.Count(i => i.LabelValue == label) / (double)InstancesIn.Count);
                }
            }
        }

        private int MaxDepth => Max(TrainingInstances[0].Features.Count, DistinctLabels.Count());
        private int MinNodeSize => Min(TrainingInstances[0].Features.Count, DistinctLabels.Count());
#pragma warning disable CS8619 // Nullability of reference types in value doesn't match target type.
        // This is suppressed because trainingInstance definitely has a non-null LabelValue. 
        private IEnumerable<string> DistinctLabels => TrainingInstances.Select(i => i.LabelValue).Distinct();
#pragma warning restore CS8619 // Nullability of reference types in value doesn't match target type.
        private Node RootNode { get; set; }

#pragma warning disable CS8618 // Non-nullable field is uninitialized.
        // This is suppressed because RootNode will be initialized by Train() which comes before any other dereferences.
        public DecisionTreeContext(List<Instance> trainingInstances) : base(trainingInstances) { }
#pragma warning restore CS8618 // Non-nullable field is uninitialized.

        private static double Xlog2X(double x) => x == 0 ? 0 : x * Log2(x);

        private static double Entropy(IEnumerable<Instance> instances)
        {
            double sum = 0;
            instances.Select(i => i.LabelValue).Distinct().ToList().ForEach(l => sum -= Xlog2X(instances.Count(i => i.LabelValue == l) / (double)instances.Count()));
            return sum;
        }

        private static double GainRatioDiscrete(List<Instance> instances, string featureName)
        {
            if (instances[0].Features[featureName].ValueType != ValueType.Discrete)
            {
                throw new ArgumentException($"Values of {featureName} is not of discrete type. ");
            }

            double sum = 0;
            instances.Select(i => i.Features[featureName].Value).Distinct().ToList().ForEach(v =>
            sum += instances.Count(i => i.Features[featureName].Value == v) / (double)instances.Count
            * Entropy(instances.Where(i => i.Features[featureName].Value == v)));

            double infoGain = Entropy(instances) - sum;

            double splitRatio = 0;
            instances.Select(i => i.Features[featureName].Value).Distinct().ToList().ForEach(v =>
            splitRatio -= Xlog2X(instances.Count(i => i.Features[featureName].Value == v) / (double)instances.Count));

            return infoGain / splitRatio;
        }

        private static double GainRatioContinuous(List<Instance> instances, string featureName, out double threshold)
        {
            if (instances[0].Features[featureName].ValueType != ValueType.Continuous)
            {
                throw new ArgumentException($"Values of {featureName} is not of continuous type. ");
            }

            List<double> distinctValues = instances.Select(i => i.Features[featureName].Value).Distinct().ToList().ConvertAll(v => (double)v);
            threshold = double.NaN;
            double maxGainRatio = 0;
            foreach (double tryThreshold in distinctValues)
            {
                List<Instance> dichotomized = new List<Instance>();
                foreach (Instance instance in instances)
                {
                    dichotomized.Add(new Instance(new Dictionary<string, Feature> { {
                        featureName, new Feature(ValueType.Discrete, instance.Features[featureName].Value <= tryThreshold ? $"less than or equal to {tryThreshold}" : $"greater than {tryThreshold}")
                    } }, instance.LabelValue, instance.LabelName));
                }
                double tryGainRatio = GainRatioDiscrete(dichotomized, featureName);
                if (tryGainRatio > maxGainRatio)
                {
                    maxGainRatio = tryGainRatio;
                    threshold = tryThreshold;
                }
            }
            return maxGainRatio;
        }

        private static string GetSplitFeature(List<Instance> instances, out double? threshold)
        {
            string featureName = "";
            double maxGainRatio = 0;
            threshold = null;
            foreach (string tryFeatureName in instances[0].Features.Select(kvp => kvp.Key))
            {
                if (instances[0].Features[tryFeatureName].ValueType == ValueType.Discrete)
                {
                    double tryGainRatio = GainRatioDiscrete(instances, tryFeatureName);
                    if (tryGainRatio > maxGainRatio)
                    {
                        maxGainRatio = tryGainRatio;
                        featureName = tryFeatureName;
                        threshold = null;
                    }
                }
                else
                {
                    double tryGainRatio = GainRatioContinuous(instances, tryFeatureName, out double tryThreshold);
                    if (tryGainRatio > maxGainRatio)
                    {
                        maxGainRatio = tryGainRatio;
                        featureName = tryFeatureName;
                        threshold = tryThreshold;
                    }
                }
            }
            return featureName;
        }

        private static Dictionary<string, List<Instance>> Split(List<Instance> instances, string featureName, double? threshold = null)
        {
            Dictionary<string, List<Instance>> splitResults = new Dictionary<string, List<Instance>>();
            if (instances[0].Features[featureName].ValueType == ValueType.Discrete)
            {
                foreach (string value in instances.Select(i => i.Features[featureName].Value).Distinct())
                {
                    splitResults.Add(value, new List<Instance>());
                }
                foreach (Instance instance in instances)
                {
                    splitResults[(string)instance.Features[featureName].Value].Add(instance);
                }
            }
            else
            {
                splitResults.Add($"less than or equal to threshold", new List<Instance>());
                splitResults.Add($"greater than threshold", new List<Instance>());
                foreach (Instance instance in instances)
                {
                    if (instance.Features[featureName].Value <= threshold)
                    {
                        splitResults["less than or equal to threshold"].Add(instance);
                    }
                    else
                    {
                        splitResults["greater than threshold"].Add(instance);
                    }
                }
            }

            return splitResults;
        }

        private void SplitRecursive(Node node, int currentDepth)
        {
            if (currentDepth >= MaxDepth || node.InstancesIn.Count <= MinNodeSize || node.InstancesIn.Select(i => i.LabelValue).Distinct().Count() == 1)
            {
                node.IsLeafNode = true;
                node.CalcLeafProbDist();
                return;
            }

            Dictionary<string, List<Instance>> branches = Split(node.InstancesIn, node.SplitFeatureName, node.SplitThreshold);
            foreach (KeyValuePair<string, List<Instance>> kvp in branches)
            {
                string subSplitFeatureName = GetSplitFeature(kvp.Value, out double? subSplitThreshold);
                node.SubNodes.Add(kvp.Key, new Node(kvp.Value, subSplitFeatureName, subSplitThreshold));
            }
            foreach (Node subNode in node.SubNodes.Select(kvp => kvp.Value))
            {
                SplitRecursive(subNode, ++currentDepth);
            }
        }

        public override void Train()
        {
            string splitFeatureName = GetSplitFeature(TrainingInstances, out double? threshold);
            RootNode = new Node(TrainingInstances, splitFeatureName, threshold);
            SplitRecursive(RootNode, 1);
        }

        public override Dictionary<string, double> GetProbDist(Instance testingInstance)
        {
            Node currentNode = RootNode;
            while (!currentNode.IsLeafNode)
            {
                currentNode = currentNode.NavigateDown(testingInstance);
            }
            return OrderedNormalized(currentNode.LeafProbDist);
        }
    }
}
