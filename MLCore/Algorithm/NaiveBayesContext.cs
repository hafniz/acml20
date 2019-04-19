using System.Collections.Generic;
using MLCore.FrameworkElement;

namespace MLCore.Algorithm
{
    public class NaiveBayesContext : AlgorithmContextBase
    {
        public NaiveBayesContext(List<Instance> trainingInstances) : base(trainingInstances) { }
        public override Dictionary<dynamic, double> GetProbDist(Instance testingInstance)
        {

        }
    }
}
