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
        private static double EuclideanDistance(Instance instance1, Instance instance2)
        {
            double distSumSquared = 0;

            foreach (string featureName in instance1.Features.Select(f => f.Name))
            {
                distSumSquared += Pow(instance1[featureName].Value - instance2[featureName].Value, 2);
            }
            return Sqrt(distSumSquared);
        }

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
            Instance[] otherInstanceArray = new Instance[TrainingInstances.Count];
            TrainingInstances.CopyTo(otherInstanceArray);
            List<Instance> otherInstances = otherInstanceArray.ToList();
            otherInstances.Remove(testingInstance);

            Dictionary<Instance, double> distStats = new Dictionary<Instance, double>();

            otherInstances.ForEach(otherInstance => distStats.Add(otherInstance, EuclideanDistance(testingInstance, otherInstance)));
            /*
            foreach (Instance otherInstance in otherInstances)
            {
                distStats.Add(otherInstance, EuclideanDistance(testingInstance, otherInstance));
            }
            */


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
            /*
            IEnumerable<Instance> neighbors = GetNeighbors(testingInstance, homoCount - 1);
            return neighbors.Count(i => i.LabelValue == testingInstance.LabelValue) / (double)homoCount;
            */
        }

        /// <summary>
        /// For experimental use. Calculates alpha values for each of the TrainingInstances. 
        /// </summary>
        /// <returns>A list of tuples consisting of each instance in TrainingInstances and its corresponding alpha value. </returns>
        public List<(Instance, double)> GetAllAlphaValues()
        {
            List<(Instance, double)> alphas = new List<(Instance, double)>();
            TrainingInstances.ForEach(trainingInstance => alphas.Add((trainingInstance, GetAlphaValue(trainingInstance))));
            /*
            foreach (Instance trainingInstance in TrainingInstances)
            {
                alphas.Add((trainingInstance, GetAlphaValue(trainingInstance)));
            }
            */
            return alphas;
        }

        public double GetBetaValue(Instance testingInstance)=> GetNeighbors(testingInstance, TrainingInstances.Count(i => i.LabelValue == testingInstance.LabelValue) - 1).Where(i => i.LabelValue == testingInstance.LabelValue).Sum(i => 1.0 / (1.0 + EuclideanDistance(i, testingInstance))) / TrainingInstances.Where(i => i != testingInstance).Sum(i => 1.0 / (1.0 + EuclideanDistance(i, testingInstance)));


        /*
        public double GetBetaValue(Instance testingInstance)
        {
            int homoCount = TrainingInstances.Count(i => i.LabelValue == testingInstance.LabelValue);
            IEnumerable<Instance> neighbors = GetNeighbors(testingInstance, homoCount - 1);
            double c = neighbors.Where(i => i.LabelValue == testingInstance.LabelValue).Sum(i => 1.0 / (1.0 + EuclideanDistance(i, testingInstance)));
            double d = TrainingInstances.Where(i => i != testingInstance).Sum(i => 1.0 / (1.0 + EuclideanDistance(i, testingInstance)));
            return c / d;
        }
        */

        public List<(Instance, double)> GetAllBetaValues()
        {
            List<(Instance, double)> betas = new List<(Instance, double)>();
            TrainingInstances.ForEach(trainingInstance=> betas.Add((trainingInstance, GetBetaValue(trainingInstance))));
            /*
            foreach (Instance trainingInstance in TrainingInstances)
            {
                betas.Add((trainingInstance, GetBetaValue(trainingInstance)));
            }*/
            return betas;
        }
    }
}
