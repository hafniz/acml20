#define ALPHA_ALLOPS
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLCore.Algorithm;

namespace MLCore
{
    static class Program
    {
        #region OALPHA_BINFREQ
#if OALPHA_BINFREQ
        static int finishedCount = 0;
        static bool hasFinished = false;
        static readonly StringBuilder logger = new StringBuilder();
        static readonly StringBuilder resultsBuilder = new StringBuilder($"filename,bin0,bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9\r\n");

        static void Main()
        {
            AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;
            //foreach (string filename in Directory.EnumerateFiles("..\\..\\..\\..\\Dataset\\UCI\\ECOC8030files\\UCI_ECOC8030"))
            //{
            //    TryGetBinFreq(filename, true);
            //}
            Parallel.ForEach(Directory.EnumerateFiles("..\\UCI_ECOC8030"), new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, filename => TryGetBinFreq(filename, true));
            logger.AppendLine($"Finished all {finishedCount}. ");
            Console.WriteLine($"Finished all {finishedCount}. ");
            Output();
            hasFinished = true;
        }

        static void CurrentDomain_ProcessExit(object? sender, EventArgs e)
        {
            if (!hasFinished)
            {
                logger.AppendLine($"Program exited after finishing {finishedCount}. ");
                Output();
            }
        }

        static void Output()
        {
            using StreamWriter resultsWriter = new StreamWriter($"..\\oalpha_binfreq.csv");
            resultsWriter.Write(resultsBuilder);
            using StreamWriter logWriter = new StreamWriter("..\\log.txt");
            logWriter.Write(logger);
        }

        static void TryGetBinFreq(string filename, bool writeDatasetWithAlpha)
        {
            List<Instance> instances = CSV.ReadFromCsv(filename, null);
            filename = Path.GetFileNameWithoutExtension(filename);
            try
            {
                double[] binFreq = new double[10];
                List<(Instance instance, double alpha)> results = new KNNContext(instances).GetAllAlphaValues();
                IEnumerable<double> alphas = results.Select(tuple => tuple.alpha);
                for (int i = 0; i < 10; i++)
                {
                    double binLowerRange = i * 0.1;
                    double binUpperRange = i == 9 ? 1.01 : binLowerRange + 0.1; // include alpha = 1.0 in bin9
                    binFreq[i] = alphas.Count(a => a >= binLowerRange && a < binUpperRange) / (double)instances.Count;
                }
                resultsBuilder.AppendLine($"{filename},{string.Join(',', binFreq)}");

                if (writeDatasetWithAlpha)
                {
                    StringBuilder sb = new StringBuilder($"{string.Join(',', instances.First().Features.Select(f => f.Name))},label,alpha\r\n");
                    foreach ((Instance instance, double alpha) in results)
                    {
                        sb.AppendLine($"{instance.Serialize()},{alpha}");
                    }
                    using StreamWriter sw = new StreamWriter($"..\\UCI_ECOC8030_withalpha\\{filename}.csv");
                    sw.Write(sb);
                }

                logger.AppendLine($"{DateTime.Now}\tSuccessfully finished {filename} (Total: {++finishedCount})");
                Console.WriteLine($"{DateTime.Now}\tSuccessfully finished {filename} (Total: {finishedCount})");
                Console.WriteLine($"{filename},{string.Join(',', binFreq)}");
            }
            catch (Exception e)
            {
                Console.WriteLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
                resultsBuilder.AppendLine($"{filename},{string.Join(',', Enumerable.Repeat("NaN", 10))}");
                logger.AppendLine(new string('>', 64));
                logger.AppendLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
                logger.AppendLine(e.ToString());
                logger.AppendLine(new string('>', 64));
            }
        }
#endif
        #endregion

        #region CALC_PROBDIST
#if CALC_PROBDIST
        static int finishedCount = 0;
        static void Main()
        {
            foreach (string filename in Directory.EnumerateFiles("..\\..\\artificial-R-all"))
            {
                TryCalcProbDist(filename);
            }
        }

        private static void TryCalcProbDist(string filename)
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
                    instances.Add(new Instance(new Dictionary<string, Feature>() { { "feature0", new Feature(ValueType.Continuous, double.Parse(fields[0])) }, { "feature1", new Feature(ValueType.Continuous, double.Parse(fields[1])) } }, labelValue: fields[2]));
                }
                DecisionTreeContext context = new DecisionTreeContext(instances);
                context.Train();
                foreach (Instance instance in instances)
                {
                    Dictionary<string, double> probDist = context.GetProbDist(instance);
                    sb.AppendLine(string.Join(',', instance.Features["feature0"].Value, instance.Features["feature1"].Value, instance.LabelValue, probDist.ContainsKey("0.0") ? probDist["0.0"] : 0.0, probDist.ContainsKey("1.0") ? probDist["1.0"] : 0.0));
                }
                using StreamWriter sw = new StreamWriter($"..\\..\\dtc44\\{Path.GetFileNameWithoutExtension(filename)}-dtc44.csv");
                sw.Write(sb);
                Console.WriteLine(++finishedCount);
            }
            catch (Exception e)
            {
                Console.WriteLine(new string('>', Console.WindowWidth - 1));
                Console.WriteLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
                Console.WriteLine(e.ToString());
                Console.WriteLine(new string('>', Console.WindowWidth - 1));
            }
        }
#endif
        #endregion

        #region XFCV_ACCURACY
#if XFCV_ACCURACY
        static readonly (Type, string) alg = (typeof(DecisionTreeContext), "dtc44");
        static readonly StringBuilder logger = new StringBuilder();
        static readonly StringBuilder resultsBuilder = new StringBuilder($"filename,{alg.Item2}-cv0,{alg.Item2}-cv1,{alg.Item2}-cv2,{alg.Item2}-cv3,{alg.Item2}-cv4,{alg.Item2}-cv5,{alg.Item2}-cv6,{alg.Item2}-cv7,{alg.Item2}-cv8,{alg.Item2}-cv9\r\n");
        static int finishedCount = 0;
        static bool hasFinished = false;

        static void Main(string[] args)
        {
            AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;
            //foreach (string filename in Directory.EnumerateFiles("..\\..\\..\\..\\Dataset\\artificial-R\\dataset2824 with alpha"))
            //{
            //    TryCvAccuracy(filename);
            //}
            int maxDegreeOfParallelism = args.Length == 0 ? 1 : (int)(Environment.ProcessorCount * double.Parse(args[0]));
            Console.WriteLine($"Max degree of parallelism: {maxDegreeOfParallelism} ");
            Parallel.ForEach(Directory.EnumerateFiles("..\\datasets"), new ParallelOptions() { MaxDegreeOfParallelism = maxDegreeOfParallelism }, filename => TryCvAccuracy(filename));
            logger.AppendLine($"Finished all {finishedCount}. ");
            Console.WriteLine($"Finished all {finishedCount}. ");
            Output();
            hasFinished = true;
        }

        static void CurrentDomain_ProcessExit(object? sender, EventArgs e)
        {
            if (!hasFinished)
            {
                logger.AppendLine($"Program exited after finishing {finishedCount}. ");
                Output();
            }
        }

        static void Output()
        {
            using StreamWriter resultsWriter = new StreamWriter($"..\\{alg.Item2}.csv");
            resultsWriter.Write(resultsBuilder);
            using StreamWriter logWriter = new StreamWriter("..\\log.txt");
            logWriter.Write(logger);
        }

        static void TryCvAccuracy(string filename)
        {
            List<Instance> instances = CSV.ReadFromCsv(filename, ..^1, hasHeader: true);
            filename = Path.GetFileNameWithoutExtension(filename);
            try
            {
                double[] accuracyValues = new double[10];
                for (int i = 0; i < 10; i++)
                {
                    CrossValidation.CvPrediction(instances, alg.Item1, out accuracyValues[i]);
                }
                resultsBuilder.AppendLine($"{filename},{string.Join(',', accuracyValues)}");
                logger.AppendLine($"{DateTime.Now}\tSuccessfully finished {filename} (Total: {++finishedCount})");
                Console.WriteLine($"{DateTime.Now}\tSuccessfully finished {filename} (Total: {finishedCount})");
                Console.WriteLine($"{filename},{string.Join(',', accuracyValues)}");
            }
            catch (Exception e)
            {
                Console.WriteLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
                resultsBuilder.AppendLine($"{filename},{string.Join(',', Enumerable.Repeat("NaN", 10))}");
                logger.AppendLine(new string('>', 64));
                logger.AppendLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
                logger.AppendLine(e.ToString());
                logger.AppendLine(new string('>', 64));
            }
        }
#endif
        #endregion

        #region ALPHA_ALLOPS
#if ALPHA_ALLOPS
        static readonly StringBuilder logger = new StringBuilder();
        static readonly StringBuilder allBinFreqBuilder = new StringBuilder("filename,knnsqrt-bin0,knnsqrt-bin1,knnsqrt-bin2,knnsqrt-bin3,knnsqrt-bin4,knnsqrt-bin5,knnsqrt-bin6,knnsqrt-bin7,knnsqrt-bin8,knnsqrt-bin9,knnallrew-bin0,knnallrew-bin1,knnallrew-bin2,knnallrew-bin3,knnallrew-bin4,knnallrew-bin5,knnallrew-bin6,knnallrew-bin7,knnallrew-bin8,knnallrew-bin9,nbpkid-bin0,nbpkid-bin1,nbpkid-bin2,nbpkid-bin3,nbpkid-bin4,nbpkid-bin5,nbpkid-bin6,nbpkid-bin7,nbpkid-bin8,nbpkid-bin9,dtc44-bin0,dtc44-bin1,dtc44-bin2,dtc44-bin3,dtc44-bin4,dtc44-bin5,dtc44-bin6,dtc44-bin7,dtc44-bin8,dtc44-bin9\r\n");
        static int finishedCount = 0;
        static bool hasFinished = false;

        static void Main(string[] args)
        {
            AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;
            //foreach (string filename in Directory.EnumerateFiles("..\\..\\..\\..\\Dataset\\UCI\\ECOC8030files\\UCI_ECOC8030"))
            //{
            //    TryAlphaAllOps(filename);
            //}
            int maxDegreeOfParallelism = (int)(Environment.ProcessorCount * (args.Length == 0 ? 1 : double.Parse(args[0])));
            Console.WriteLine($"Max degree of parallelism: {maxDegreeOfParallelism} ");
            Parallel.ForEach(Directory.EnumerateFiles("..\\datasets"), new ParallelOptions() { MaxDegreeOfParallelism = maxDegreeOfParallelism }, filename => TryAlphaAllOps(filename));
            logger.AppendLine($"Finished all {finishedCount}. ");
            Console.WriteLine($"Finished all {finishedCount}. ");
            Output();
            hasFinished = true;
        }

        static void TryAlphaAllOps(string filename)
        {
            try
            {
                // 1. read dataset with alpha
                Dictionary<Instance, Dictionary<string, double>> datasetInfo = new Dictionary<Instance, Dictionary<string, double>>();
                List<Instance> instances = CSV.ReadFromCsv(filename, ..^1, null, true);
                instances.ForEach(i => datasetInfo.Add(i, new Dictionary<string, double>()));
                List<double> alphas = CSV.ReadFromCsv(filename, ^1.., true).SelectColumn(0).ConvertAll(s => double.Parse(s));
                int temp = 0;
                foreach (KeyValuePair<Instance, Dictionary<string, double>> kvp in datasetInfo)
                {
                    kvp.Value.Add("alpha", alphas[temp++]);
                }
                filename = Path.GetFileNameWithoutExtension(filename);
                StringBuilder fileBinFreqBuilder = new StringBuilder($"{filename},");

                // 2. do work
                foreach ((AlgorithmContextBase context, string symbol) in new List<(AlgorithmContextBase context, string symbol)>
                {
                    (new KNNContext(instances) { NeighboringMethod = KNNContext.NeighboringOption.SqrtNeighbors }, "knnsqrt"),
                    (new KNNContext(instances) { NeighboringMethod = KNNContext.NeighboringOption.AllNeighborsWithReweighting }, "knnallrew"),
                    (new NaiveBayesContext(instances), "nbpkid"),
                    (new DecisionTreeContext(instances) { UseLaplaceCorrection = true }, "dtc44")
                })
                {
                    // 2.1 calc prob dist
                    context.Train();
                    foreach (Instance instance in instances)
                    {
                        Dictionary<string, double> result = context.GetProbDist(instance);
                        double p0 = result.ContainsKey("0.0") ? result["0.0"] : 0.0;
                        double p1 = result.ContainsKey("1.0") ? result["1.0"] : 0.0;
                        bool isCorrect = true;
                        if (p0 > p1)
                        {
                            isCorrect = instance.LabelValue == "0.0";
                        }
                        else if (p1 > p0)
                        {
                            isCorrect = instance.LabelValue == "1.0";
                        }
                        datasetInfo[instance].Add($"{symbol}-p0", p0);
                        datasetInfo[instance].Add($"{symbol}-p1", p1);
                        datasetInfo[instance].Add($"{symbol}-iscorrect", isCorrect ? 1.0 : 0.0);
                    }

                    // 2.2 calc alpha
                    List<Instance> derivedInstances = new List<Instance>();
                    foreach (Instance instance in instances)
                    {
                        derivedInstances.Add(new Instance(new List<Feature>
                        {
                            new Feature($"{symbol}-p0", ValueType.Continuous, datasetInfo[instance][$"{symbol}-p0"]),
                            new Feature($"{symbol}-p1", ValueType.Continuous, datasetInfo[instance][$"{symbol}-p1"]),
                        }, instance.LabelValue));
                    }

                    List<double> derivedAlphas = new KNNContext(derivedInstances).GetAllAlphaValues().Select(tuple => tuple.Item2).ToList();
                    temp = 0;
                    foreach (KeyValuePair<Instance, Dictionary<string, double>> kvp in datasetInfo)
                    {
                        kvp.Value.Add($"{symbol}-alpha", derivedAlphas[temp]);
                        kvp.Value.Add($"{symbol}-adiff", derivedAlphas[temp++] - kvp.Value["alpha"]);
                    }

                    // 2.3 record bin freq
                    for (int i = 0; i < 10; i++)
                    {
                        double binLowerBound = i / 10.0;
                        double binUpperBound = i == 9 ? 1.01 : (i + 1) / 10.0;
                        fileBinFreqBuilder.Append(derivedAlphas.Count(a => a < binUpperBound && a >= binLowerBound) / (double)instances.Count);
                        fileBinFreqBuilder.Append(',');
                    }
                }
                allBinFreqBuilder.AppendLine(fileBinFreqBuilder.ToString()[..^1]);

                // 3. write dataset with alphas 
                List<List<string>> tableFields = new List<List<string>>();
                foreach ((Instance instance, Dictionary<string, double> props) in datasetInfo)
                {
                    List<string> rowFields = instance.Serialize().Split(',').ToList();
                    foreach (KeyValuePair<string, double> kvp in props)
                    {
                        rowFields.Add(kvp.Value.ToString());
                    }
                    tableFields.Add(rowFields);
                }
                CSV.WriteToCsv($"..\\datasets with alpha\\{filename}.csv", new Table<string>(tableFields), $"{string.Join(',', instances.First().Features.Select(f => f.Name))},label,{string.Join(',', datasetInfo.First().Value.Select(kvp => kvp.Key))}");

                logger.AppendLine($"{DateTime.Now}\tSuccessfully finished {filename} (Total: {++finishedCount})");
                Console.WriteLine($"{DateTime.Now}\tSuccessfully finished {filename} (Total: {finishedCount})");
            }
            catch (Exception e)
            {
                Console.WriteLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
                logger.AppendLine(new string('>', 64));
                logger.AppendLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
                logger.AppendLine(e.ToString());
                logger.AppendLine(new string('>', 64));
            }
        }

        static void CurrentDomain_ProcessExit(object? sender, EventArgs e)
        {
            if (!hasFinished)
            {
                logger.AppendLine($"Program exited after finishing {finishedCount}. ");
                Output();
            }
        }

        static void Output()
        {
            using StreamWriter allBinFreqWriter = new StreamWriter("..\\datasets-allBinFreq.csv");
            allBinFreqWriter.Write(allBinFreqBuilder);
            using StreamWriter logWriter = new StreamWriter("..\\log.txt");
            logWriter.Write(logger);
        }
#endif
        #endregion
    }
}
