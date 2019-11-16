using System;
using System.Collections.Generic;
using System.Linq;
using MLCore.Algorithm;

namespace MLCore
{
    public static class CrossValidation
    {
        /// <summary>
        /// Get the probability distribution of each instance in a dataset using cross validation. 
        /// </summary>
        /// <param name="instances">List of instances in the dataset. </param>
        /// <param name="algorithm">Type of the algorithm to be used. </param>
        /// <param name="fold">Number of fold in the cross validation. The default value is 10. </param>
        /// <returns>A dictionary mapping each instance in the dataset to a tuple consisting of probability distribution and the number of fold the instance is in. </returns>
        public static Dictionary<Instance, (Dictionary<string, double>, int)> CVProbDist(List<Instance> instances, Type algorithm, int fold = 10)
        {
            Random random = new Random();
            Dictionary<int, List<Instance>> folds = new Dictionary<int, List<Instance>>();
            int foldSize = instances.Count / fold;
            for (int i = 0; i < fold; i++)
            {
                folds.Add(i, new List<Instance>());
                for (int j = 0; j < foldSize; j++)
                {
                    int instanceNumber = random.Next(instances.Count);
                    folds[i].Add(instances[instanceNumber]);
                    instances.RemoveAt(instanceNumber);
                }
            }

            //         instance              label    prob    fold
            Dictionary<Instance, (Dictionary<string, double>, int)> results = new Dictionary<Instance, (Dictionary<string, double>, int)>();
            for (int testingFoldNumber = 0; testingFoldNumber < fold; testingFoldNumber++)
            {
                List<Instance> trainingInstances = new List<Instance>();
                foreach (List<Instance> trainingFold in folds.Where(kvp => kvp.Key != testingFoldNumber).Select(kvp => kvp.Value))
                {
                    trainingInstances = trainingInstances.Concat(trainingFold).ToList();
                }
                List<Instance> trainingInstancesCloned = trainingInstances.Select(i => (Instance)i.Clone()).ToList();
                AlgorithmContextBase context;
                context = (AlgorithmContextBase)(Activator.CreateInstance(algorithm, trainingInstancesCloned) ?? throw new NullReferenceException("Failed to create instance of algorithm context. "));
                context.Train();
                foreach (Instance testingInstance in folds[testingFoldNumber])
                {
                    results.Add(testingInstance, (context.GetProbDist((Instance)testingInstance.Clone()), testingFoldNumber));
                }
            }
            return results.OrderBy(kvp => kvp.Value.Item2).ThenBy(kvp => kvp.Key.LabelValue).ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }
    }
}
