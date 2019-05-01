using System.Collections.Generic;
using System.Linq;

namespace MLCore.Algorithm
{
    public abstract class AlgorithmContextBase
    {
        protected List<Instance> TrainingInstances { get; set; }
        protected AlgorithmContextBase(List<Instance> trainingInstances) => TrainingInstances = trainingInstances;
        public virtual void Train() { } // Default implementation for algorithms that need not training, e.g., kNN.
        public abstract Dictionary<string, double> GetProbDist(Instance testingInstance);
        public string Classify(Instance testingInstance) => GetProbDist(testingInstance).ToArray()[0].Key;
    }
}
