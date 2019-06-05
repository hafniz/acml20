using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MLCore.Algorithm;

namespace MLCore
{
    static class Program
    {
        static int finishedCount = 0;
        static void Main()
        {
            /*
            //STEP 1: Generate 10 * 5 = 50 linear separated datasets.
            List<Instance> testTemplate = CSV.ReadFromCsv("testTemplate.csv", null);
            for (int i = 2; i <= 20; i += 2)
            {
                foreach (char c in new List<char>() { 'A', 'B', 'V', 'G', 'D' })
                {
                    CSV.WriteToCsv($".\\Output\\LS{(i < 10 ? "0" : "") + i.ToString()}{c}.csv", Generation.GenerateLinearSeperated(i, testTemplate, null, $".\\angles\\LS{(i < 10 ? "0" : "") + i.ToString()}{c}.txt"));
                }
            }

            //STEP 2: Sample 50 * 4 = 200 sets of points.
            foreach (string filename in Directory.EnumerateFiles(".\\Output\\levelNeg1-dataset", "*", SearchOption.AllDirectories))
            {
                for (int i = 1; i <= 4; i++)
                {
                    int seed = new Random().Next();
                    Random random = new Random(seed);
                    File.WriteAllText($"{filename[0..^4]}{i}.txt", seed.ToString());
                    List<List<string>> rows = CSV.ReadFromCsv(filename, false);
                    List<List<string>> samples = new List<List<string>>();
                    for (int j = 0; j < 250; j++)
                    {
                        samples.Add(rows[random.Next(2500)]);
                    }
                    CSV.WriteToCsv($"{filename[0..^4]}{i}.csv", samples);
                }
            }

            //STEP 3.1: Generate 200 * 2(KNN, NB) = 400 "LEVEL 0" datasets
            foreach (string filename in Directory.EnumerateFiles(".\\Output\\sample-dataset", "*", SearchOption.AllDirectories))
            {
                GenerateDatasetKNNNB(filename, ".\\Output\\testTemplate.csv");
                Console.WriteLine($"Finished {filename}");
            }

            void GenerateDatasetKNNNB(string trainFile, string testTemplate)
            {
                Dictionary<string, Type> algorithms = new Dictionary<string, Type>() { { "KNN", typeof(KNNContext) }, { "NB", typeof(NaiveBayesContext) } };
                foreach (KeyValuePair<string, Type> algorithm in algorithms)
                {
                    List<Instance> trainingInstances = CSV.ReadFromCsv(trainFile, null);
                    AlgorithmContextBase context = (AlgorithmContextBase)Activator.CreateInstance(algorithm.Value, trainingInstances);
                    context.Train();
                    List<Instance> testingInstances = CSV.ReadFromCsv(testTemplate, null);
                    List<Instance> predictResults = new List<Instance>();
                    foreach (Instance testingInstance in testingInstances)
                    {
                        string predictLabel = context.Classify(testingInstance);
                        Instance predictInstance = new Instance(testingInstance.Features, predictLabel, testingInstance.LabelName);
                        predictResults.Add(predictInstance);
                    }
                    CSV.WriteToCsv($".\\Output\\level0-dataset\\{Path.GetFileNameWithoutExtension(trainFile)}-{algorithm.Key}.csv", predictResults);
                }
            }

            //STEP 3.2: Generate 200 * 1(DT) = 200 "LEVEL 0" datasets
            foreach (string filename in Directory.EnumerateFiles(".\\Output\\sample-dataset", "*", SearchOption.AllDirectories))
            {
                GenerateDatasetDT(filename, ".\\Output\\testTemplate.csv");
                Console.WriteLine($"Finished {filename}");
            }

            void GenerateDatasetDT(string trainFile, string testTemplate)
            {
                int targetTreeDepth = int.Parse(trainFile[^8..^6]) / 2;
                List<Instance> trainingInstances = CSV.ReadFromCsv(trainFile, null);
                DecisionTreeContext context = new DecisionTreeContext(trainingInstances, targetTreeDepth);
                context.Train();
                List<Instance> testingInstances = CSV.ReadFromCsv(testTemplate, null);
                List<Instance> predictResults = new List<Instance>();
                foreach (Instance testingInstance in testingInstances)
                {
                    string predictLabel = context.Classify(testingInstance);
                    Instance predictInstance = new Instance(testingInstance.Features, predictLabel, testingInstance.LabelName);
                    predictResults.Add(predictInstance);
                }
                CSV.WriteToCsv($".\\Output\\level0-dataset\\{Path.GetFileNameWithoutExtension(trainFile)}-DT.csv", predictResults);
            }

            //STEP 3.3: Generate 200 * 1(RT) = 200 "LEVEL 0" datasets
            for (int i = 2; i <= 20; i += 2)
            {
                for (int j = 1; j <= 20; j++)
                {
                    string name = $"RT{(i < 10 ? "0" : "") + i.ToString()}-{(j < 10 ? "0" : "") + j.ToString()}";
                    DecisionTreeContext.Node tree = DecisionTreeContext.GenerateTree(i / 2, ($".\\Output\\testTemplate.csv", $".\\Output\\level0\\level0-dataset\\{name}.csv"));
                    File.WriteAllText($".\\Output\\level0\\level0-RTstructure\\{name}.txt", tree.ToString());
                    Console.WriteLine($"Finished {name}");
                }
            }

            //STEP 4: Calculate alpha values for the 400 + 200 + 200 = 800 "LEVEL 0" datasets
            //List<string> filenames = Directory.EnumerateFiles(".\\Output\\level0\\level0-dataset").ToList();
            //Task[] tasks = new Task[8];
            //for (int i = 0; i < 8; i++)
            //{
            //    List<string> sublist = filenames.GetRange(i * 100, 100);
            //    tasks[i] = Task.Run(() => sublist.ForEach(n => CalculateAlpha(n)));
            //}
            //Task.WaitAll(tasks);
            //Console.WriteLine("Finished all. ");

            //static void CalculateAlpha(string filename)
            //{
            //    List<Instance> instances = CSV.ReadFromCsv(filename, null);
            //    Dictionary<Instance, string> resultsSerialized = new Dictionary<Instance, string>();
            //    List<(Instance, double)> alphas = new KNNContext(instances).GetAllAlphaValues().ToList();
            //    foreach ((Instance instance, double alphaValue) in alphas)
            //    {
            //        resultsSerialized[instance] = alphaValue.ToString();
            //    }
            //    List<List<string>> dataToWrite = new List<List<string>>();
            //    foreach (KeyValuePair<Instance, string> kvp in resultsSerialized)
            //    {
            //        List<string> row = kvp.Key.Serialize().Split(',').ToList();
            //        dataToWrite.Add(row.Concat(new List<string>() { kvp.Value }).ToList());
            //    }
            //    CSV.WriteToCsv(filename, dataToWrite, "feature0,feature1,label,alpha");
            //    Console.WriteLine($"Finished {filename}");
            //}
            
            //// STEP 5: Compute probDist for the 800 datasets
            //List<(string abbr, Type type, string header)> algorithmInfo = new List<(string abbr, Type type, string header)>()
            //{
            //    ("KNN", typeof(KNNContext), "feature0,feature1,label,alpha,knn-cv0-fold,knn-cv0-p0,knn-cv0-p1,knn-cv1-fold,knn-cv1-p0,knn-cv1-p1,knn-cv2-fold,knn-cv2-p0,knn-cv2-p1,knn-cv3-fold,knn-cv3-p0,knn-cv3-p1,knn-cv4-fold,knn-cv4-p0,knn-cv4-p1,knn-cv5-fold,knn-cv5-p0,knn-cv5-p1,knn-cv6-fold,knn-cv6-p0,knn-cv6-p1,knn-cv7-fold,knn-cv7-p0,knn-cv7-p1,knn-cv8-fold,knn-cv8-p0,knn-cv8-p1,knn-cv9-fold,knn-cv9-p0,knn-cv9-p1"),
            //    ("NB", typeof(NaiveBayesContext), "feature0,feature1,label,alpha,nb-cv0-fold,nb-cv0-p0,nb-cv0-p1,nb-cv1-fold,nb-cv1-p0,nb-cv1-p1,nb-cv2-fold,nb-cv2-p0,nb-cv2-p1,nb-cv3-fold,nb-cv3-p0,nb-cv3-p1,nb-cv4-fold,nb-cv4-p0,nb-cv4-p1,nb-cv5-fold,nb-cv5-p0,nb-cv5-p1,nb-cv6-fold,nb-cv6-p0,nb-cv6-p1,nb-cv7-fold,nb-cv7-p0,nb-cv7-p1,nb-cv8-fold,nb-cv8-p0,nb-cv8-p1,nb-cv9-fold,nb-cv9-p0,nb-cv9-p1"),
            //    ("DT", typeof(DecisionTreeContext), "feature0,feature1,label,alpha,dt-cv0-fold,dt-cv0-p0,dt-cv0-p1,dt-cv1-fold,dt-cv1-p0,dt-cv1-p1,dt-cv2-fold,dt-cv2-p0,dt-cv2-p1,dt-cv3-fold,dt-cv3-p0,dt-cv3-p1,dt-cv4-fold,dt-cv4-p0,dt-cv4-p1,dt-cv5-fold,dt-cv5-p0,dt-cv5-p1,dt-cv6-fold,dt-cv6-p0,dt-cv6-p1,dt-cv7-fold,dt-cv7-p0,dt-cv7-p1,dt-cv8-fold,dt-cv8-p0,dt-cv8-p1,dt-cv9-fold,dt-cv9-p0,dt-cv9-p1")
            //};

            //List<string> filenames = Directory.EnumerateFiles(".\\Output\\level0\\level0-dataset").ToList();
            //Task[] tasks = new Task[8];
            //for (int i = 0; i < 8; i++)
            //{
            //    List<string> sublist = filenames.GetRange(i * 100, 100);
            //    tasks[i] = Task.Run(() => sublist.ForEach(s => ComputeProbDist(algorithmInfo, s)));
            //}
            //Task.WaitAll(tasks);
            //Console.WriteLine("Finished all. ");

            //static void ComputeProbDist(List<(string abbr, Type type, string header)> algorithmInfo, string filename)
            //{
            //    List<List<string>> rawData = CSV.ReadFromCsv(filename, true);
            //    List<Instance> instances = new List<Instance>();
            //    rawData.ForEach(l => instances.Add(new Instance(new Dictionary<string, Feature>()
            //    {
            //        { "feature0", new Feature(ValueType.Continuous, double.Parse(l[0]))},
            //        { "feature1", new Feature(ValueType.Continuous, double.Parse(l[1]))}
            //    }, l[2])));

            //    foreach ((string abbr, Type type, string header) in algorithmInfo)
            //    {
            //        Dictionary<Instance, List<string>> resultsSerialized = new Dictionary<Instance, List<string>>();
            //        instances.ForEach(i => resultsSerialized[i] = new List<string>());
            //        for (int i = 0; i < 10; i++)
            //        {
            //            Instance[] instancesCopy = new Instance[instances.Count];
            //            instances.CopyTo(instancesCopy);
            //            int targetTreeDepth = int.Parse(Path.GetFileNameWithoutExtension(filename).Substring(2, 2));
            //            Dictionary<Instance, (Dictionary<string, double>, int)> cvResults = CrossValidation.CVProbDist(instancesCopy.ToList(), type, targetTreeDepth);

            //            foreach (KeyValuePair<Instance, (Dictionary<string, double>, int)> kvp in cvResults)
            //            {
            //                resultsSerialized[kvp.Key].Add(kvp.Value.Item2.ToString());
            //                resultsSerialized[kvp.Key].Add(kvp.Value.Item1.ContainsKey("0.0") ? kvp.Value.Item1["0.0"].ToString() : "0.0");
            //                resultsSerialized[kvp.Key].Add(kvp.Value.Item1.ContainsKey("1.0") ? kvp.Value.Item1["1.0"].ToString() : "0.0");
            //            }
            //        }

            //        List<List<string>> dataToWrite = new List<List<string>>();
            //        for (int i = 0; i < resultsSerialized.Count; i++)
            //        {
            //            KeyValuePair<Instance, List<string>> kvp = resultsSerialized.ToList()[i];
            //            List<string> row = kvp.Key.Serialize().Split(',').ToList();
            //            dataToWrite.Add(row.Concat(new List<string>() { rawData[i][^1] }).Concat(kvp.Value).ToList());
            //        }

            //        string outputFilename = $".\\Output\\level1\\level1-CVresults\\{Path.GetFileNameWithoutExtension(filename)}-{abbr}.csv";
            //        CSV.WriteToCsv(outputFilename, dataToWrite, header);
            //        Console.WriteLine($"{DateTime.Now} Finished {outputFilename} ");
            //    }
            //}
            */
            // STEP 6: Calculate average for 10 sets of xfcv results, compute alpha and alpha shift
            List<string> filenames = Directory.EnumerateFiles(".\\Output\\level0\\level0-dataset").ToList();
            Task[] tasks = new Task[8];
            for (int i = 0; i < 8; i++)
            {
                List<string> sublist = filenames.GetRange(i * 100, 100);
                tasks[i] = Task.Run(() => sublist.ForEach(s => WriteSummary(s)));
            }
            Task.WaitAll(tasks);
            Console.WriteLine("Finished all. ");

            static void WriteSummary(string filename)
            {
                // init and props
                const string headers = "feature0,feature1,label,alpha,knn-avg-p0,knn-avg-p1,knn-avg-alpha,knn-alphashift,nb-avg-p0,nb-avg-p1,nb-avg-alpha,nb-alphashift,dt-avg-p0,dt-avg-p1,dt-avg-alpha,dt-alphashift";
                const int labelColumnIndex = 2;
                const int origAlphaColumnIndex = 3;
                IEnumerable<List<string>> summaryColumns = CSV.ReadFromCsv(filename, true).Transpose();
                List<int> p0ColumnIndexes = new List<int>() { 5, 8, 11, 14, 17, 20, 23, 26, 29, 32 };

                // for each algorithm
                foreach (string algorithmAbbr in new List<string>() { "KNN", "NB", "DT" })
                {
                    string cvResultsFilename = $".\\Output\\level1\\level1-CVresults\\{Path.GetFileNameWithoutExtension(filename)}-{algorithmAbbr}.csv";
                    List<List<string>> cvResultsTable = CSV.ReadFromCsv(cvResultsFilename, true);
                    List<List<double>> tenAlphaColumns = new List<List<double>>();

                    // for each xfcv
                    foreach (int p0ColumnIndex in p0ColumnIndexes)
                    {
                        List<string> p0Column = cvResultsTable.SelectColumn(p0ColumnIndex);
                        List<string> p1Column = cvResultsTable.SelectColumn(p0ColumnIndex + 1);
                        List<string> labelColumn = cvResultsTable.SelectColumn(labelColumnIndex);
                        List<List<string>> joinedTable = TabularOperation.JoinColumns(p0Column, p1Column, labelColumn);

                        // assume ordered
                        IEnumerable<Instance> derivedInstances = new List<Instance>();
                        foreach (List<string> row in joinedTable)
                        {
                            // deserialize and create Instance
                            Dictionary<string, Feature> derivedFeatures = new Dictionary<string, Feature>
                            {
                                { "p0Feature", new Feature(ValueType.Continuous, double.Parse(row[0])) },
                                { "p1Feature", new Feature(ValueType.Continuous, double.Parse(row[1])) }
                            };
                            derivedInstances = derivedInstances.Append(new Instance(derivedFeatures, row[^1], "label"));
                        }

                        KNNContext context = new KNNContext(derivedInstances.ToList());
                        // assume ordered as original instances
                        List<(Instance instance, double alphaValue)> alphas = context.GetAllAlphaValues();
                        tenAlphaColumns.Add(alphas.Select(t => t.alphaValue).ToList());
                    }

                    // calculate average for 10 xfcv results for this algorithm
                    List<List<string>>[] joinedPTables = new List<List<string>>[10];
                    for (int i = 0; i < 10; i++)
                    {
                        joinedPTables[i] = TabularOperation.JoinColumns(cvResultsTable.SelectColumn(p0ColumnIndexes[i]), cvResultsTable.SelectColumn(p0ColumnIndexes[i] + 1));
                    }
                    List<List<string>> pAverageColumns = TabularOperation.Average(joinedPTables).Transpose();
                    summaryColumns = summaryColumns.Append(pAverageColumns[0]);
                    summaryColumns = summaryColumns.Append(pAverageColumns[1]);

                    // calculate average alpha
                    List<List<string>>[] joinedAlphaTables = new List<List<string>>[10];
                    for (int i = 0; i < 10; i++)
                    {
                        joinedAlphaTables[i] = TabularOperation.JoinColumns(tenAlphaColumns[i].ConvertAll(d => d.ToString()));
                    }
                    List<string> alphaAverageColumn = TabularOperation.Average(joinedAlphaTables).Transpose().Single();
                    summaryColumns = summaryColumns.Append(alphaAverageColumn);

                    // calculate alpha shift
                    List<string> alphaShiftColumn = alphaAverageColumn.Minus(summaryColumns.ToList()[origAlphaColumnIndex]);
                    summaryColumns = summaryColumns.Append(alphaShiftColumn);
                }

                // write summary into file
                string summaryFilename = $".\\Output\\level1\\level1-summary\\{Path.GetFileNameWithoutExtension(filename)}-SMRY.csv";
                CSV.WriteToCsv(summaryFilename, summaryColumns.ToList().Transpose(), headers);
                Console.WriteLine($"{DateTime.Now} Finished {summaryFilename} ({++finishedCount}/800)");
            }
        }
    }
}
