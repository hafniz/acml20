using System.Collections.Generic;
using System.Linq;

namespace MLCore.Algorithm
{
    public abstract class AlgorithmContextBase
    {
        protected List<Instance> TrainingInstances { get; }
        protected AlgorithmContextBase(List<Instance> trainingInstances) => TrainingInstances = trainingInstances;
        public virtual void Train() { } // Default implementation for algorithms that need not training, e.g., kNN.
        public abstract Dictionary<string, double> GetProbDist(Instance testingInstance);
        public string Classify(Instance testingInstance) => GetProbDist(testingInstance).First().Key;
        protected static Dictionary<string, double> OrderedNormalized(Dictionary<string, double> dict)
        {
            double sum = dict.Select(kvp => kvp.Value).Sum();
            Dictionary<string, double> normalized = new Dictionary<string, double>();
            foreach (KeyValuePair<string, double> kvp in dict)
            {
                normalized.Add(kvp.Key, kvp.Value / sum);
            }
            return normalized.OrderByDescending(kvp => kvp.Value).ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }
    }
}
