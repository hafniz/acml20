using System;
using System.Collections.Generic;
using System.Linq;
using MLCore.Algorithm;

namespace MLCore
{
    /// <summary>
    /// Entry point of the application. Parses the input arguments, updates RuntimeConfig, 
    /// calls other functions with Runtime Config and outputs messages to the command line.
    /// </summary>
    static class Program
    {
        static void Main(string[] args)
        {
#if DEBUG
#else
            Console.WriteLine($"{DateTime.Now}  Computing in progress... ");
            string header = "feature0,feature1,label,alpha," +
                "knn-cv0-fold,knn-cv0-p0,knn-cv0-p1,knn-cv1-fold,knn-cv1-p0,knn-cv1-p1,knn-cv2-fold,knn-cv2-p0,knn-cv2-p1,knn-cv3-fold,knn-cv3-p0,knn-cv3-p1,knn-cv4-fold,knn-cv4-p0,knn-cv4-p1,knn-cv5-fold,knn-cv5-p0,knn-cv5-p1,knn-cv6-fold,knn-cv6-p0,knn-cv6-p1,knn-cv7-fold,knn-cv7-p0,knn-cv7-p1,knn-cv8-fold,knn-cv8-p0,knn-cv8-p1,knn-cv9-fold,knn-cv9-p0,knn-cv9-p1," +
                "nb-cv0-fold,nb-cv0-p0,nb-cv0-p1,nb-cv1-fold,nb-cv1-p0,nb-cv1-p1,nb-cv2-fold,nb-cv2-p0,nb-cv2-p1,nb-cv3-fold,nb-cv3-p0,nb-cv3-p1,nb-cv4-fold,nb-cv4-p0,nb-cv4-p1,nb-cv5-fold,nb-cv5-p0,nb-cv5-p1,nb-cv6-fold,nb-cv6-p0,nb-cv6-p1,nb-cv7-fold,nb-cv7-p0,nb-cv7-p1,nb-cv8-fold,nb-cv8-p0,nb-cv8-p1,nb-cv9-fold,nb-cv9-p0,nb-cv9-p1," +
                "dt-cv0-fold,dt-cv0-p0,dt-cv0-p1,dt-cv1-fold,dt-cv1-p0,dt-cv1-p1,dt-cv2-fold,dt-cv2-p0,dt-cv2-p1,dt-cv3-fold,dt-cv3-p0,dt-cv3-p1,dt-cv4-fold,dt-cv4-p0,dt-cv4-p1,dt-cv5-fold,dt-cv5-p0,dt-cv5-p1,dt-cv6-fold,dt-cv6-p0,dt-cv6-p1,dt-cv7-fold,dt-cv7-p0,dt-cv7-p1,dt-cv8-fold,dt-cv8-p0,dt-cv8-p1,dt-cv9-fold,dt-cv9-p0,dt-cv9-p1";

            List<Type> algorithms = new List<Type>() { typeof(KNNContext), typeof(NaiveBayesContext), typeof(DecisionTreeContext) };
            for (int datasetNumber = int.Parse(args[0]); datasetNumber <= int.Parse(args[1]); datasetNumber++)
            {
                List<Instance> instances = CSV.ReadFromCsv($".\\artificial\\{datasetNumber}.txt", null);
                Dictionary<Instance, List<string>> resultsSerialized = new Dictionary<Instance, List<string>>();
                instances.ForEach(i => resultsSerialized[i] = new List<string>());

                List<(Instance, double)> alphas = new KNNContext(instances).GetAllAlphaValues().ToList();
                foreach ((Instance, double) tuple in alphas)
                {
                    resultsSerialized[tuple.Item1].Add(tuple.Item2.ToString());
                }

                foreach (Type algorithm in algorithms)
                {
                    for (int j = 0; j < 10; j++)
                    {
                        Instance[] instancesCopy = new Instance[instances.Count];
                        instances.CopyTo(instancesCopy);
                        Dictionary<Instance, (Dictionary<string, double>, int)> cvResults = CrossValidation.CVProbDist(instancesCopy.ToList(), algorithm);
                        foreach (KeyValuePair<Instance, (Dictionary<string, double>, int)> kvp in cvResults)
                        {
                            resultsSerialized[kvp.Key].Add(kvp.Value.Item2.ToString());
                            resultsSerialized[kvp.Key].Add(kvp.Value.Item1.ContainsKey("0.0") ? kvp.Value.Item1["0.0"].ToString() : "0");
                            resultsSerialized[kvp.Key].Add(kvp.Value.Item1.ContainsKey("1.0") ? kvp.Value.Item1["1.0"].ToString() : "0");
                        }
                    }
                }

                List<List<string>> dataToWrite = new List<List<string>>();
                foreach (KeyValuePair<Instance, List<string>> kvp in resultsSerialized)
                {
                    List<string> row = kvp.Key.Serialize().Split(',').ToList();
                    dataToWrite.Add(row.Concat(kvp.Value).ToList());
                }
                CSV.WriteToCsv($".\\results\\{datasetNumber}_results.csv", dataToWrite, header: header);
                Console.WriteLine($"{DateTime.Now}  Finished computing on dataset #{datasetNumber}. ");
            }
#endif
        }
    }
}
