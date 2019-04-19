using System.Collections.Generic;
using System.Linq;
using MLCore.FrameworkElement;

namespace MLCore.Algorithm
{
    public abstract class AlgorithmContextBase
    {
        protected List<Instance> TrainingInstances { get; set; }
        protected AlgorithmContextBase(List<Instance> trainingInstances) => TrainingInstances = trainingInstances;
        public virtual void Train() { } // Default implementation for algorithms that need not training, e.g., kNN.
        public abstract Dictionary<dynamic, double> GetProbDist(Instance testingInstance);
        public dynamic Classify(Instance testingInstance) => GetProbDist(testingInstance).OrderByDescending(kvp => kvp.Value).ToArray()[0].Key;
    }
}
