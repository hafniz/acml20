using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
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

        /// <summary>
        /// Archived from Program.cs, region CALC_PROBDIST. Calculates probDist of algorithm dtc44 for dataset in Dataset\\artificial-R. To calculate probDist of other algorithms, change the code accordingly. 
        /// </summary>
        /// <param name="filename"></param>
        public static void TryCalcProbDist(string filename)
        {
            try
            {
                StringBuilder sb = new StringBuilder($"feature0,feature1,label,dtc44-p0,dtc44-p1\r\n");
                List<Instance> instances = new List<Instance>();
                foreach (string row in File.ReadLines(filename))
                {
                    if (row.StartsWith("feature0"))
                    {
                        continue;
                    }
                    string[] fields = row.Split(',');
                    instances.Add(new Instance(new List<Feature>() { new Feature("feature0", ValueType.Continuous, double.Parse(fields[0])), new Feature("feature1", ValueType.Continuous, double.Parse(fields[1])) }, labelValue: fields[2]));
                }
                DecisionTreeContext context = new DecisionTreeContext(instances);
                context.Train();
                foreach (Instance instance in instances)
                {
                    Dictionary<string, double> probDist = context.GetProbDist(instance);
                    sb.AppendLine(string.Join(',', instance["feature0"].Value, instance["feature1"].Value, instance.LabelValue, probDist.ContainsKey("0.0") ? probDist["0.0"] : 0.0, probDist.ContainsKey("1.0") ? probDist["1.0"] : 0.0));
                }
                using StreamWriter sw = new StreamWriter($"..\\..\\dtc44\\{Path.GetFileNameWithoutExtension(filename)}-dtc44.csv");
                sw.Write(sb);
            }
            catch (Exception e)
            {
                Console.WriteLine(new string('>', Console.WindowWidth - 1));
                Console.WriteLine($"{DateTime.Now}\t{e.GetType()} encountered in processing {filename}, skipping this file");
                Console.WriteLine(e.ToString());
                Console.WriteLine(new string('>', Console.WindowWidth - 1));
            }
        }

        /// <summary>
        /// Archived from Program.cs, region XFCV_ACCURACY. Calculates prediction accuracy of 10-fold cross validation for all three algorithms.
        /// </summary>
        public static void TryCvAccuracy(string filename)
        {
            List<Instance> instances = CSV.ReadFromCsv(filename, null);
            filename = Path.GetFileNameWithoutExtension(filename);
            try
            {
                string s = $"{filename}";
                foreach (Type type in new Type[] { typeof(KNNContext), typeof(NaiveBayesContext), typeof(DecisionTreeContext) })
                {
                    double[] accuracyValues = new double[10];
                    for (int i = 0; i < 10; i++)
                    {
                        CvPrediction(instances, type, out accuracyValues[i]);
                    }
                    s += $",{string.Join(',', accuracyValues)}";
                }
                File.WriteAllText($"..\\xfcv-results\\{filename}.csv", s);
                Console.WriteLine($"{DateTime.Now}\tSuccessfully finished {filename}");
            }
            catch (Exception e)
            {
                Console.WriteLine(new string('>', 64));
                Console.WriteLine($"{DateTime.Now}\t{e.GetType()} encountered in processing {filename}, skipping this file");
                Console.WriteLine(e.ToString());
                Console.WriteLine(new string('>', 64));
            }
        }
    }
}
