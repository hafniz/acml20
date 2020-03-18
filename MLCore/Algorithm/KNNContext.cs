using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using static System.Math;

namespace MLCore.Algorithm
{
    public class KNNContext : AlgorithmContextBase
    {
        public enum NeighboringOption
        {
            AllNeighbors,
            AllNeighborsWithReweighting,
            SqrtNeighbors
        }

        public NeighboringOption NeighboringMethod { get; set; } = NeighboringOption.AllNeighborsWithReweighting;
        public KNNContext(List<Instance> trainingInstances) : base(trainingInstances) { }

        [DebuggerStepThrough]
        private static double EuclideanDistance(Instance instance1, Instance instance2) => Sqrt(instance1.Features.Sum(f => (double)Pow(f.Value - instance2[f.Name].Value, 2)));

        public override Dictionary<string, double> GetProbDist(Instance testingInstance)
        {
            Dictionary<string, double> distStats = new Dictionary<string, double>();
            foreach (Instance neighborInstance in GetNeighbors(testingInstance, NeighboringMethod == NeighboringOption.SqrtNeighbors ? (int)Sqrt(TrainingInstances.Count) : TrainingInstances.Count - 1))
            {
                if (neighborInstance.LabelValue is null)
                {
                    throw new NullReferenceException("Unlabeled instance is used as training instance. ");
                }
                if (!distStats.ContainsKey(neighborInstance.LabelValue))
                {
                    distStats.Add(neighborInstance.LabelValue, 0);
                }
                distStats[neighborInstance.LabelValue] += EuclideanDistance(neighborInstance, testingInstance);
            }

            Dictionary<string, double> distStatsInverted = new Dictionary<string, double>();
            foreach (KeyValuePair<string, double> kvp in distStats)
            {
                distStatsInverted.Add(kvp.Key, 1.0 / kvp.Value * (NeighboringMethod == NeighboringOption.AllNeighborsWithReweighting ? TrainingInstances.Count(i => i.LabelValue == kvp.Key) / (double)TrainingInstances.Count : 1.0));
            }
            return OrderedNormalized(distStatsInverted);
        }

        /// <summary>
        /// Get k nearest neighbors of an instance. 
        /// </summary>
        /// <param name="testingInstance">The instance to be used as a reference based on which distances to all other instances are calculated. </param>
        /// <param name="k">Number of neighbors to be selected. </param>
        /// <returns>Instances that are nearest to the testingInstance by Euclidean distance. </returns>
        private IEnumerable<Instance> GetNeighbors(Instance testingInstance, int k)
        {
            Dictionary<Instance, double> distStats = new Dictionary<Instance, double>();
            TrainingInstances.ForEach(i => distStats.Add(i, EuclideanDistance(testingInstance, i)));
            distStats.Remove(testingInstance);
            distStats = distStats.OrderBy(kvp => kvp.Value).Take(k).ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
            foreach (KeyValuePair<Instance, double> kvp in distStats)
            {
                yield return kvp.Key;
            }
        }

        /// <summary>
        /// For experimental use. Alpha measures the ratio of agreeing neighbors. In this case every instance in the TrainingInstances is considered a neighbour. 
        /// </summary>
        /// <param name="testingInstance">The instance to be calculated alpha value on. </param>
        /// <returns>The value of alpha. </returns>
        public double GetAlphaValue(Instance testingInstance)
        {
            int homoCount = TrainingInstances.Count(i => i.LabelValue == testingInstance.LabelValue);
            return GetNeighbors(testingInstance, homoCount - 1).Count(i => i.LabelValue == testingInstance.LabelValue) / (double)homoCount;
        }

        /// <summary>
        /// For experimental use. Calculates alpha values for each of the TrainingInstances. 
        /// </summary>
        /// <returns>A list of tuples consisting of each instance in TrainingInstances and its corresponding alpha value. </returns>
        public IEnumerable<(Instance, double)> GetAllAlphaValues()
        {
            foreach (Instance trainingInstance in TrainingInstances)
            {
                yield return (trainingInstance, GetAlphaValue(trainingInstance));
            }
        }

        public double GetBetaValue(Instance testingInstance) =>
            GetNeighbors(testingInstance, TrainingInstances.Count(i => i.LabelValue == testingInstance.LabelValue) - 1)
            .Where(i => i.LabelValue == testingInstance.LabelValue).Sum(i => 1.0 / (1.0 + EuclideanDistance(i, testingInstance)))
            / TrainingInstances.Where(i => i != testingInstance).Sum(i => 1.0 / (1.0 + EuclideanDistance(i, testingInstance)));

        public IEnumerable<(Instance, double)> GetAllBetaValues()
        {
            Dictionary<string, int> homoCount = new Dictionary<string, int>();
            TrainingInstances.ForEach(i =>
            {
                if (!homoCount.ContainsKey(i.LabelValue ?? throw new NullReferenceException("Cannot compute beta value for an unlabeled instance. ")))
                {
                    homoCount.Add(i.LabelValue, TrainingInstances.Count(instance => instance.LabelValue == i.LabelValue));
                }
            });

            double[,] distStats = new double[TrainingInstances.Count, TrainingInstances.Count];
            for (int i = 1; i < TrainingInstances.Count; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    distStats[i, j] = distStats[j, i] = EuclideanDistance(TrainingInstances[i], TrainingInstances[j]);
                }
            }

            for (int i = 0; i < TrainingInstances.Count; i++)
            {
                Instance currInstance = TrainingInstances[i];
                Dictionary<Instance, double> distToOtherInstances = new Dictionary<Instance, double>(Enumerable.Range(0, TrainingInstances.Count).Select(j => new KeyValuePair<Instance, double>(TrainingInstances[j], distStats[i, j])));
                distToOtherInstances.Remove(TrainingInstances[i]);
                yield return (currInstance,
                    distToOtherInstances.OrderBy(kvp => kvp.Value).Take(homoCount[currInstance.LabelValue ?? throw new NullReferenceException("Cannot compute beta value for an unlabeled instance. ")] - 1)
                    .Where(kvp => kvp.Key.LabelValue == currInstance.LabelValue).Sum(kvp => 1.0 / (1.0 + kvp.Value))
                    / distToOtherInstances.Sum(kvp => 1.0 / (1.0 + kvp.Value)));
            }
        }
    }
}
