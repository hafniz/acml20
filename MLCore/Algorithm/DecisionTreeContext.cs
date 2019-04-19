using System.Collections.Generic;
using MLCore.FrameworkElement;

namespace MLCore.Algorithm
{
    public class DecisionTreeContext : AlgorithmContextBase
    {
        public DecisionTreeContext(List<Instance> trainingInstances) : base(trainingInstances) { }
        public override Dictionary<dynamic, double> GetProbDist(Instance testingInstance)
        {

        }
    }
}
