using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using static System.Math;

namespace MLCore.Algorithm
{
    public class KNNContext : AlgorithmContextBase
    {
        public KNNContext(List<Instance> trainingInstances) : base(trainingInstances) { }
        [DebuggerStepThrough]
        private static double EuclideanDistance(Instance instance1, Instance instance2)
        {
            double distSumSquared = 0;
            foreach (string featureName in instance1.Features.Select(kvp => kvp.Key))
            {
                distSumSquared += Pow(instance1.Features[featureName].Value - instance2.Features[featureName].Value, 2);
            }
            return Sqrt(distSumSquared);
        }

        public override Dictionary<string, double> GetProbDist(Instance testingInstance)
        {
            Dictionary<string, double> distStats = new Dictionary<string, double>();
            foreach (Instance trainingInstance in TrainingInstances)
            {
#pragma warning disable CS8604 // Possible null reference argument. 
                // This is suppressed because trainingInstance definitely has a non-null LabelValue. 
                if (!distStats.ContainsKey(trainingInstance.LabelValue))
                {
                    distStats.Add(trainingInstance.LabelValue, 0);
                }
                distStats[trainingInstance.LabelValue] += EuclideanDistance(trainingInstance, testingInstance);
#pragma warning restore CS8604 // Possible null reference argument.
            }

            Dictionary<string, double> distStatsNormalized = new Dictionary<string, double>();
            foreach (KeyValuePair<string, double> kvp in distStats)
            {
                distStatsNormalized.Add(kvp.Key, kvp.Value / TrainingInstances.Count(i => i.LabelValue == kvp.Key));
            }

            Dictionary<string, double> distStatsInverted = new Dictionary<string, double>();
            foreach (KeyValuePair<string, double> kvp in distStatsNormalized)
            {
                distStatsInverted.Add(kvp.Key, 1 / kvp.Value);
            }
            return OrderedNormalized(distStatsInverted);
        }
        public double GetAlphaValue(Instance testingInstance)
        {
            IEnumerable<Instance> GetNeighbors(Instance testingInstance, int k)
            {
                Instance[] otherInstanceArray = new Instance[TrainingInstances.Count];
                TrainingInstances.CopyTo(otherInstanceArray);
                List<Instance> otherInstances = otherInstanceArray.ToList();
                otherInstances.Remove(testingInstance);
                Dictionary<Instance, double> distStats = new Dictionary<Instance, double>();
                foreach (Instance otherInstance in otherInstances)
                {
                    distStats.Add(otherInstance, EuclideanDistance(testingInstance, otherInstance));
                }
                distStats = distStats.OrderBy(kvp => kvp.Value).Take(k).ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
                foreach (KeyValuePair<Instance, double> kvp in distStats)
                {
                    yield return kvp.Key;
                }
            }

            int homoCount = TrainingInstances.Count(i => i.LabelValue == testingInstance.LabelValue);
            List<Instance> neighbors = GetNeighbors(testingInstance, homoCount - 1).ToList();
            return neighbors.Count(i => i.LabelValue == testingInstance.LabelValue) / (double)homoCount;
        }
        public List<(Instance, double)> GetAllAlphaValues()
        {
            IEnumerable<(Instance, double)> alphas = new List<(Instance, double)>();
            foreach (Instance trainingInstance in TrainingInstances)
            {
                alphas = alphas.Append((trainingInstance, GetAlphaValue(trainingInstance)));
            }
            return alphas.ToList();
        }
    }
}
