using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MLCore.Algorithm;

namespace MLCore
{
    public static class CrossValidation
    {
        [DebuggerStepThrough]
        private static Dictionary<int, List<Instance>> Fold(List<Instance> instances, int foldCount)
        {
            Random random = new Random();
            Dictionary<int, List<Instance>> folds = new Dictionary<int, List<Instance>>();
            Instance[] instancesArray = new Instance[instances.Count];
            instances.CopyTo(instancesArray);
            List<Instance> instancesCopy = instancesArray.ToList();
            int foldSize = instancesCopy.Count / foldCount;
            int reminder = instancesCopy.Count - foldSize * foldCount;
            for (int i = 0; i < foldCount; i++)
            {
                folds.Add(i, new List<Instance>());
                for (int j = 0; j < (i < reminder ? foldSize + 1 : foldSize); j++)
                {
                    int instanceNumber = random.Next(instancesCopy.Count);
                    folds[i].Add(instancesCopy[instanceNumber]);
                    instancesCopy.RemoveAt(instanceNumber);
                }
            }
            return folds;
        }

        /// <summary>
        /// Get the probability distribution of each instance in a dataset using cross validation. 
        /// </summary>
        /// <param name="instances">List of instances in the dataset. </param>
        /// <param name="algorithm">Type of the algorithm to be used. </param>
        /// <param name="foldCount">Number of fold in the cross validation. The default value is 10. </param>
        /// <returns>A dictionary mapping each instance in the dataset to a tuple consisting of probability distribution and the number of fold the instance is in. </returns>
        public static Dictionary<Instance, (Dictionary<string, double>, int)> CvProbDist(List<Instance> instances, Type algorithm, int foldCount = 10)
        {
            Dictionary<int, List<Instance>> folds = Fold(instances, foldCount);
            Dictionary<Instance, (Dictionary<string, double> probDist, int testingFoldNumber)> results = new Dictionary<Instance, (Dictionary<string, double>, int)>();
            for (int testingFoldNumber = 0; testingFoldNumber < foldCount; testingFoldNumber++)
            {
                IEnumerable<Instance> trainingInstances = new List<Instance>();
                foreach (List<Instance> trainingFold in folds.Where(kvp => kvp.Key != testingFoldNumber).Select(kvp => kvp.Value))
                {
                    trainingInstances = trainingInstances.Concat(trainingFold);
                }
                IEnumerable<Instance> trainingInstancesCloned = trainingInstances.Select(i => (Instance)i.Clone());
                AlgorithmContextBase context = (AlgorithmContextBase)(Activator.CreateInstance(algorithm, trainingInstancesCloned.ToList()) ?? throw new NullReferenceException("Failed to create instance of algorithm context. "));
                context.Train();
                foreach (Instance testingInstance in folds[testingFoldNumber])
                {
                    results.Add(testingInstance, (context.GetProbDist((Instance)testingInstance.Clone()), testingFoldNumber));
                }
            }
            return results;
        }

        public static Dictionary<Instance, (string, int)> CvPrediction(List<Instance> instances, Type algorithm, out double accuracy, int foldCount = 10)
        {
            Dictionary<int, List<Instance>> folds = Fold(instances, foldCount);
            Dictionary<Instance, (string prediction, int testingFoldNumber)> results = new Dictionary<Instance, (string, int)>();
            for (int testingFoldNumber = 0; testingFoldNumber < foldCount; testingFoldNumber++)
            {
                IEnumerable<Instance> trainingInstances = new List<Instance>();
                foreach (List<Instance> trainingFold in folds.Where(kvp => kvp.Key != testingFoldNumber).Select(kvp => kvp.Value))
                {
                    trainingInstances = trainingInstances.Concat(trainingFold);
                }
                IEnumerable<Instance> trainingInstancesCloned = trainingInstances.Select(i => (Instance)i.Clone());
                AlgorithmContextBase context = (AlgorithmContextBase)(Activator.CreateInstance(algorithm, trainingInstancesCloned.ToList()) ?? throw new NullReferenceException("Failed to create instance of algorithm context. "));
                context.Train();
                foreach (Instance testingInstance in folds[testingFoldNumber])
                {
                    results.Add(testingInstance, (context.Classify((Instance)testingInstance.Clone()), testingFoldNumber));
                }
            }
            accuracy = results.Count(kvp => kvp.Value.prediction == kvp.Key.LabelValue) / (double)instances.Count;
            return results;
        }
    }
}
