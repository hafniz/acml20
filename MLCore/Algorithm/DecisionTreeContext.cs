using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using static System.Math;

namespace MLCore.Algorithm
{
    public class DecisionTreeContext : AlgorithmContextBase
    {
        public class Node
        {
            public string SplitFeatureName { get; set; }
            public double SplitThreshold { get; set; }
            public bool IsLeafNode { get; set; }
            public List<Instance> InstancesIn { get; }

            //            labelValue, probability
            public Dictionary<string, double> LeafProbDist { get; set; } = new Dictionary<string, double>();

            //          featureValue, subNode
            public Dictionary<string, Node> SubNodes { get; } = new Dictionary<string, Node>();

            public Node(List<Instance> instancesIn, string splitFeatureName, double splitThreshold)
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
                if (instance[SplitFeatureName].ValueType == ValueType.Discrete)
                {
                    return SubNodes[instance[SplitFeatureName].Value];
                }
                if (instance[SplitFeatureName].Value <= SplitThreshold)
                {
                    return SubNodes["less than or equal to threshold"];
                }
                return SubNodes["greater than threshold"];
            }

            // Provide priorProb <=> use Laplace correction; leave priorProb as null <=> do not use Laplace correction
            public void CalcLeafProbDist(Dictionary<string, double>? priorProb)
            {
                foreach (string? label in priorProb?.Select(kvp => kvp.Key) ?? InstancesIn.Select(i => i.LabelValue).Distinct())
                {
                    LeafProbDist.Add(label ?? throw new NullReferenceException("Unlabeled instances used in growing a tree. "), priorProb is null
                        ? InstancesIn.Count(i => i.LabelValue == label) / (double)InstancesIn.Count
                        : (InstancesIn.Count(i => i.LabelValue == label) + priorProb[label]) / (InstancesIn.Count + priorProb.Sum(kvp => kvp.Value)));
                }
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder();
                AppendNodeContent(this, 0);
                return sb.ToString();

                void AppendNodeContent(Node currentNode, int currentDepth)
                {
                    if (currentNode.IsLeafNode)
                    {
                        foreach (KeyValuePair<string, double> kvp in currentNode.LeafProbDist)
                        {
                            sb.AppendLine($"{new string(' ', currentDepth * 4)}{kvp.Key}: {kvp.Value}");
                        }
                        return;
                    }
                    if (currentNode.SplitThreshold == double.NaN) // Discrete
                    {
                        foreach (KeyValuePair<string, Node> branch in currentNode.SubNodes)
                        {
                            sb.AppendLine($"{new string(' ', currentDepth * 4)}{currentNode.SplitFeatureName} = {branch.Key}: ");
                            AppendNodeContent(branch.Value, currentDepth + 1);
                        }
                    }
                    else // Continuous
                    {
                        sb.AppendLine($"{new string(' ', currentDepth * 4)}{currentNode.SplitFeatureName} <= {currentNode.SplitThreshold}: ");
                        AppendNodeContent(currentNode.SubNodes["less than or equal to threshold"], currentDepth + 1);
                        sb.AppendLine($"{new string(' ', currentDepth * 4)}{currentNode.SplitFeatureName} > {currentNode.SplitThreshold}: ");
                        AppendNodeContent(currentNode.SubNodes["greater than threshold"], currentDepth + 1);
                    }
                }
            }
        }

        private Dictionary<string, double>? priorProb = null;
        private Node? rootNode;
        public bool UseLaplaceCorrection { get; set; } = true;
        public DecisionTreeContext(List<Instance> trainingInstances) : base(trainingInstances) { }

        [DebuggerStepThrough]
        private static double Xlog2X(double x) => x == 0 ? 0 : x * Log2(x);

        [DebuggerStepThrough]
        private static double Entropy(IEnumerable<Instance> instances)
        {
            double sum = 0;
            instances.Select(i => i.LabelValue).Distinct().ToList().ForEach(l => sum -= Xlog2X(instances.Count(i => i.LabelValue == l) / (double)instances.Count()));
            return sum;
        }

        private static double GainRatioDiscrete(List<Instance> instances, string featureName)
        {
            if (instances.First()[featureName].ValueType != ValueType.Discrete)
            {
                throw new ArgumentException($"Values of {featureName} is not of discrete type. ");
            }

            double sum = 0;
            instances.Select(i => i[featureName].Value).Distinct().ToList().ForEach(v =>
            sum += instances.Count(i => i[featureName].Value == v) / (double)instances.Count * Entropy(instances.Where(i => i[featureName].Value == v)));
            double infoGain = Entropy(instances) - sum;
            double splitRatio = 0;
            instances.Select(i => i[featureName].Value).Distinct().ToList().ForEach(v =>
            splitRatio -= Xlog2X(instances.Count(i => i[featureName].Value == v) / (double)instances.Count));
            return infoGain / splitRatio;
        }

        // If not successful, return value will be 0 and out threshold will be double.NaN
        private static double GainRatioContinuous(List<Instance> instances, string featureName, out double threshold)
        {
            if (instances.First()[featureName].ValueType != ValueType.Continuous)
            {
                throw new ArgumentException($"Values of {featureName} is not of continuous type. ");
            }

            List<double> distinctValues = instances.Select(i => i[featureName].Value).Distinct().ToList().ConvertAll(v => (double)v);
            threshold = double.NaN;
            double maxGainRatio = 0;
            foreach (double tryThreshold in distinctValues)
            {
                List<Instance> dichotomized = new List<Instance>();
                foreach (Instance instance in instances)
                {
                    dichotomized.Add(new Instance(new List<Feature> {
                        new Feature(featureName, ValueType.Discrete, instance[featureName].Value <= tryThreshold ? $"less than or equal to {tryThreshold}" : $"greater than {tryThreshold}")
                    }, instance.LabelValue, instance.LabelName));
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

        // If not successful, return value will be string.Empty and out threshold will be double.NaN
        private static string GetSplitFeature(List<Instance> instances, out double threshold)
        {
            string featureName = string.Empty;
            double maxGainRatio = 0;
            threshold = double.NaN;
            foreach (string tryFeatureName in instances.First().Features.Select(f => f.Name))
            {
                if (instances.First()[tryFeatureName].ValueType == ValueType.Discrete)
                {
                    double tryGainRatio = GainRatioDiscrete(instances, tryFeatureName);
                    if (tryGainRatio > maxGainRatio)
                    {
                        maxGainRatio = tryGainRatio;
                        featureName = tryFeatureName;
                        threshold = double.NaN;
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

        private static Dictionary<string, List<Instance>> Split(List<Instance> instances, string featureName, double threshold = double.NaN)
        {
            //      featureValue, instances
            Dictionary<string, List<Instance>> subGroups = new Dictionary<string, List<Instance>>();
            if (instances.First()[featureName].ValueType == ValueType.Discrete)
            {
                foreach (string featureValue in instances.Select(i => i[featureName].Value).Distinct())
                {
                    subGroups.Add(featureValue, new List<Instance>());
                }
                foreach (Instance instance in instances)
                {
                    subGroups[(string)instance[featureName].Value].Add(instance);
                }
            }
            else
            {
                subGroups.Add("less than or equal to threshold", new List<Instance>());
                subGroups.Add("greater than threshold", new List<Instance>());
                foreach (Instance instance in instances)
                {
                    if (instance[featureName].Value <= threshold)
                    {
                        subGroups["less than or equal to threshold"].Add(instance);
                    }
                    else
                    {
                        subGroups["greater than threshold"].Add(instance);
                    }
                }
            }
            return subGroups;
        }

        private void SplitRecursive(Node node, int currentDepth)
        {
            bool IsBaseCase() => node.InstancesIn.Count <= 1 // No more than 1 instance in the node
               || node.InstancesIn.Select(i => i.LabelValue).Distinct().Count() <= 1 // All instances in the node are class-homogeneous
               || string.IsNullOrEmpty(node.SplitFeatureName); // None of the features provide any gain ratio (information gain)

            currentDepth++;
            if (IsBaseCase())
            {
                node.IsLeafNode = true;
                node.CalcLeafProbDist(priorProb);
                return;
            }

            Dictionary<string, List<Instance>> branches = Split(node.InstancesIn, node.SplitFeatureName, node.SplitThreshold);
            foreach (KeyValuePair<string, List<Instance>> kvp in branches)
            {
                string subSplitFeatureName = GetSplitFeature(kvp.Value, out double subSplitThreshold);
                node.SubNodes.Add(kvp.Key, new Node(kvp.Value, subSplitFeatureName, subSplitThreshold));
            }
            foreach (Node subNode in node.SubNodes.Select(kvp => kvp.Value))
            {
                SplitRecursive(subNode, currentDepth);
            }
        }

        public override void Train()
        {
            if (UseLaplaceCorrection)
            {
                priorProb = new Dictionary<string, double>();
                IEnumerable<string?> distinctLabels = TrainingInstances.Select(i => i.LabelValue).Distinct();
                foreach (string? labelValue in distinctLabels)
                {
                    priorProb.Add(labelValue ?? throw new NullReferenceException("Unlabeled instance in training instances. "), TrainingInstances.Count(i => i.LabelValue == labelValue) / (double)TrainingInstances.Count);
                }
            }
            string splitFeatureName = GetSplitFeature(TrainingInstances, out double threshold);
            rootNode = new Node(TrainingInstances, splitFeatureName, threshold);
            SplitRecursive(rootNode, -1);
        }

        public override Dictionary<string, double> GetProbDist(Instance testingInstance)
        {
            Node? currentNode = rootNode;
            if (currentNode is null)
            {
                throw new NullReferenceException("Root node is null. ");
            }
            while (!currentNode.IsLeafNode)
            {
                currentNode = currentNode.NavigateDown(testingInstance);
            }
            return OrderedNormalized(currentNode.LeafProbDist);
        }

        public override string ToString() => rootNode?.ToString() ?? "(Tree with null root node)";

        /// <summary>
        /// For experimental use. Randomly generates rules of a 2-dimensional (values of both features are continuous), binary decision tree and a binary-labeled dataset classified by the tree. 
        /// </summary>
        /// <param name="maxDepth">Target depth of the tree. </param>
        /// <param name="outputConfig">The filenames, including extensions, of the file containing instances to be tested on and location where test results are to be saved respectively. If left null, testing and saving will not be performed. </param>
        /// <returns>The root node of the generated tree. </returns>
        public static Node GenerateRtTree(int maxDepth, (string testTemplate, string outputFilename)? outputConfig = null)
        {
            (Node node, List<(string splitFeatureName, double leftBound, double rightBound)> bounds) rootNodeInfo;
            rootNodeInfo.node = new Node(new List<Instance>(), "", double.NaN);
            rootNodeInfo.bounds = new List<(string splitFeatureName, double leftBound, double rightBound)> { ("feature0", 0, 1), ("feature1", 0, 1) };
            int startingFeatureIndex = new Random().Next(2);
            SplitFeatures(rootNodeInfo, maxDepth, -1, startingFeatureIndex);

            if (!(outputConfig is null))
            {
                List<Instance> testingInstances = CSV.ReadFromCsv(outputConfig.Value.testTemplate, null);
                List<Instance> predictResults = new List<Instance>();
                foreach (Instance testingInstance in testingInstances)
                {
                    Node currentNode = rootNodeInfo.node;
                    while (!currentNode.IsLeafNode)
                    {
                        currentNode = currentNode.NavigateDown(testingInstance);
                    }
                    predictResults.Add(new Instance(testingInstance.Features, currentNode.LeafProbDist.Single(kvp => kvp.Value == 1).Key));
                }
                CSV.WriteToCsv(outputConfig.Value.outputFilename, predictResults);
            }
            return rootNodeInfo.node;

            static void SplitFeatures((Node node, List<(string splitFeatureName, double leftBound, double rightBound)> bounds) nodeInfo, int maxDepth, int currentDepth, int splitFeatureIndex)
            {
                Random random = new Random();
                currentDepth++;

                nodeInfo.node.SplitFeatureName = nodeInfo.bounds[splitFeatureIndex].splitFeatureName;

                // This limits the splitThreshold within range of 40% - 60% between leftBound and rightBound. Can be changed or deleted according to demand. 
                double leftBound = nodeInfo.bounds[splitFeatureIndex].leftBound;
                double rightBound = nodeInfo.bounds[splitFeatureIndex].rightBound;
                double offset = random.NextDouble(); //0.4 + random.NextDouble() / 5;
                double splitThreshold = leftBound + offset * (rightBound - leftBound);
                nodeInfo.node.SplitThreshold = splitThreshold;

                nodeInfo.node.SubNodes.Add("less than or equal to threshold", new Node(new List<Instance>(), "", double.NaN));
                nodeInfo.node.SubNodes.Add("greater than threshold", new Node(new List<Instance>(), "", double.NaN));

                if (currentDepth + 1 >= maxDepth) // subnodes of current node are leaf nodes
                {
                    nodeInfo.node.SubNodes["less than or equal to threshold"].IsLeafNode = true;
                    nodeInfo.node.SubNodes["greater than threshold"].IsLeafNode = true;
                    if (new Random().Next() % 2 == 1) // first subnode is positive
                    {
                        nodeInfo.node.SubNodes["less than or equal to threshold"].LeafProbDist = new Dictionary<string, double>() { { "1.0", 1 }, { "0.0", 0 } };
                        nodeInfo.node.SubNodes["greater than threshold"].LeafProbDist = new Dictionary<string, double>() { { "0.0", 1 }, { "1.0", 0 } };
                    }
                    else
                    {
                        nodeInfo.node.SubNodes["less than or equal to threshold"].LeafProbDist = new Dictionary<string, double>() { { "0.0", 1 }, { "1.0", 0 } };
                        nodeInfo.node.SubNodes["greater than threshold"].LeafProbDist = new Dictionary<string, double>() { { "1.0", 1 }, { "0.0", 0 } };
                    }
                    return;
                }

                List<(string splitFeatureName, double leftBound, double rightBound)> lowBounds;
                List<(string splitFeatureName, double leftBound, double rightBound)> highBounds;

                if (splitFeatureIndex == 0)
                {
                    lowBounds = new List<(string splitFeatureName, double leftBound, double rightBound)>()
                    {
                        ("feature0", nodeInfo.bounds[0].leftBound, splitThreshold), ("feature1", nodeInfo.bounds[1].leftBound, nodeInfo.bounds[1].rightBound)
                    };
                    highBounds = new List<(string splitFeatureName, double leftBound, double rightBound)>()
                    {
                        ("feature0", splitThreshold, nodeInfo.bounds[0].rightBound), ("feature1", nodeInfo.bounds[1].leftBound, nodeInfo.bounds[1].rightBound)
                    };
                }
                else
                {
                    lowBounds = new List<(string splitFeatureName, double leftBound, double rightBound)>()
                    {
                        ("feature0", nodeInfo.bounds[0].leftBound, nodeInfo.bounds[0].rightBound), ("feature1", nodeInfo.bounds[1].leftBound, splitThreshold)
                    };
                    highBounds = new List<(string splitFeatureName, double leftBound, double rightBound)>()
                    {
                        ("feature0", nodeInfo.bounds[1].leftBound, nodeInfo.bounds[0].rightBound), ("feature1", splitThreshold, nodeInfo.bounds[1].rightBound)
                    };
                }

                SplitFeatures((nodeInfo.node.SubNodes["less than or equal to threshold"], lowBounds), maxDepth, currentDepth, splitFeatureIndex == 0 ? 1 : 0);
                SplitFeatures((nodeInfo.node.SubNodes["greater than threshold"], highBounds), maxDepth, currentDepth, splitFeatureIndex == 0 ? 1 : 0);
            }
        }
    }
}
