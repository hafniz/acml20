using System.Collections.Generic;
using MLCore.FrameworkElement;

namespace MLCore.Algorithm
{
    public class KNNContext : AlgorithmContextBase
    {
        public KNNContext(List<Instance> trainingInstances) : base(trainingInstances) { }
        public override Dictionary<dynamic, double> GetProbDist(Instance testingInstance)
        {

        }
    }
}
