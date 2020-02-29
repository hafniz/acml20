#define BETA_EXPR
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
        static readonly StringBuilder logger = new StringBuilder();
        static int finishedCount = 0;
        static bool hasFinished = false;

        static void Main(string[] args)
        {
            AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;
            //foreach (string filename in Directory.EnumerateFiles($"{Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)}\\source\\repos\\MachineLearning\\Dataset\\a270\\A270RERES\\A270RERES-new"))
            //{
            //    TryCvAccuracy(filename);
            //}
            int maxDegreeOfParallelism = (int)(Environment.ProcessorCount * double.Parse(args[0]));
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
            using StreamWriter logWriter = new StreamWriter("..\\log.txt");
            logWriter.Write(logger);
        }

        static void TryCvAccuracy(string filename)
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
                        CrossValidation.CvPrediction(instances, type, out accuracyValues[i]);
                    }
                    s += $",{string.Join(',', accuracyValues)}";
                }
                File.WriteAllText($"..\\xfcv-results\\{filename}.csv", s);
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
#endif
        #endregion

        #region ALPHA_ALLOPS
#if ALPHA_ALLOPS
        static readonly StringBuilder logger = new StringBuilder();
        static readonly StringBuilder allBinFreqBuilder = new StringBuilder("filename,oalpha-bin0,oalpha-bin1,oalpha-bin2,oalpha-bin3,oalpha-bin4,oalpha-bin5,oalpha-bin6,oalpha-bin7,oalpha-bin8,oalpha-bin9,knnallrew-bin0,knnallrew-bin1,knnallrew-bin2,knnallrew-bin3,knnallrew-bin4,knnallrew-bin5,knnallrew-bin6,knnallrew-bin7,knnallrew-bin8,knnallrew-bin9,nbpkid-bin0,nbpkid-bin1,nbpkid-bin2,nbpkid-bin3,nbpkid-bin4,nbpkid-bin5,nbpkid-bin6,nbpkid-bin7,nbpkid-bin8,nbpkid-bin9,dtc44-bin0,dtc44-bin1,dtc44-bin2,dtc44-bin3,dtc44-bin4,dtc44-bin5,dtc44-bin6,dtc44-bin7,dtc44-bin8,dtc44-bin9,knnallrew-adiff-bin0,knnallrew-adiff-bin1,knnallrew-adiff-bin2,knnallrew-adiff-bin3,knnallrew-adiff-bin4,knnallrew-adiff-bin5,knnallrew-adiff-bin6,knnallrew-adiff-bin7,knnallrew-adiff-bin8,knnallrew-adiff-bin9,nbpkid-adiff-bin0,nbpkid-adiff-bin1,nbpkid-adiff-bin2,nbpkid-adiff-bin3,nbpkid-adiff-bin4,nbpkid-adiff-bin5,nbpkid-adiff-bin6,nbpkid-adiff-bin7,nbpkid-adiff-bin8,nbpkid-adiff-bin9,dtc44-adiff-bin0,dtc44-adiff-bin1,dtc44-adiff-bin2,dtc44-adiff-bin3,dtc44-adiff-bin4,dtc44-adiff-bin5,dtc44-adiff-bin6,dtc44-adiff-bin7,dtc44-adiff-bin8,dtc44-adiff-bin9\r\n");
        static int finishedCount = 0;
        static bool hasFinished = false;

        static void Main(string[] args)
        {
            Directory.SetCurrentDirectory($"{Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)}\\source\\repos\\MachineLearning\\Dataset\\B739\\B739-noAlphas");
            AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;
            //foreach (string filename in Directory.EnumerateFiles(".\\"))
            //{
            //    TryAlphaAllOps(filename);
            //}
            int maxDegreeOfParallelism = (int)(Environment.ProcessorCount * double.Parse(args[0]));
            Console.WriteLine($"Max degree of parallelism: {maxDegreeOfParallelism} ");
            Parallel.ForEach(Directory.EnumerateFiles(".\\"), new ParallelOptions() { MaxDegreeOfParallelism = maxDegreeOfParallelism }, filename => TryAlphaAllOps(filename));
            logger.AppendLine($"Finished all {finishedCount}. ");
            Console.WriteLine($"Finished all {finishedCount}. ");
            Output();
            hasFinished = true;
        }

        static void TryAlphaAllOps(string filename)
        {
            try
            {
                // 1. read raw datasets
                Dictionary<Instance, Dictionary<string, double>> datasetInfo = new Dictionary<Instance, Dictionary<string, double>>();
                List<Instance> instances = CSV.ReadFromCsv(filename, null);
                instances.ForEach(i => datasetInfo.Add(i, new Dictionary<string, double>()));

                foreach ((Instance instance, double alpha) in new KNNContext(instances).GetAllAlphaValues())
                {
                    datasetInfo[instance].Add("alpha", alpha);
                }

                filename = Path.GetFileNameWithoutExtension(filename);
                //StringBuilder fileBinFreqBuilder = new StringBuilder($"{filename},");

                // 2. do work
                foreach ((AlgorithmContextBase context, string symbol) in new List<(AlgorithmContextBase context, string symbol)>
                {
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
                        double p0 = result.ContainsKey("0") ? result["0"] : 0.0;
                        double p1 = result.ContainsKey("1") ? result["1"] : 0.0;
                        bool isCorrect = true;
                        if (p0 > p1)
                        {
                            isCorrect = instance.LabelValue == "0";
                        }
                        else if (p1 > p0)
                        {
                            isCorrect = instance.LabelValue == "1";
                        }
                        datasetInfo[instance].Add($"{symbol}-p0", p0);
                        datasetInfo[instance].Add($"{symbol}-p1", p1);
                        datasetInfo[instance].Add($"{symbol}-iscorrect", isCorrect ? 1 : 0);
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
                    int temp = 0;
                    foreach (KeyValuePair<Instance, Dictionary<string, double>> kvp in datasetInfo)
                    {
                        kvp.Value.Add($"{symbol}-alpha", derivedAlphas[temp]);
                        kvp.Value.Add($"{symbol}-adiff", derivedAlphas[temp++] - kvp.Value["alpha"]);
                    }

                    // 2.3 record bin freq
                    //for (int i = 0; i < 10; i++)
                    //{
                    //    double binLowerBound = i / 10.0;
                    //    double binUpperBound = i == 9 ? 1.01 : (i + 1) / 10.0;
                    //    fileBinFreqBuilder.Append(derivedAlphas.Count(a => a < binUpperBound && a >= binLowerBound) / (double)instances.Count);
                    //    fileBinFreqBuilder.Append(',');
                    //}
                }
                //allBinFreqBuilder.AppendLine(fileBinFreqBuilder.ToString()[..^1]);

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
                CSV.WriteToCsv($"..\\B739-allAlphas\\{filename}.csv", new Table<string>(tableFields), $"{string.Join(',', instances.First().Features.Select(f => f.Name))},label,{string.Join(',', datasetInfo.First().Value.Select(kvp => kvp.Key))}");

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
            using StreamWriter allBinFreqWriter = new StreamWriter("..\\B739-allBinFreqs.csv");
            allBinFreqWriter.Write(allBinFreqBuilder);
            using StreamWriter logWriter = new StreamWriter("..\\log.txt");
            logWriter.Write(logger);
        }
#endif
        #endregion

        #region A270
#if A270
        static readonly StringBuilder logger = new StringBuilder();
        static int finishedCount = 0;
        static bool hasFinished = false;

        static void Main()
        {
            AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;
            //foreach (string filename in Directory.EnumerateFiles("..\\a270-datasets"))
            //{
            //    TryGetBaseAlphas(filename);
            //}
            Parallel.ForEach(Directory.EnumerateFiles("..\\a270-datasets"), filename => TryGetBaseAlphas(filename));
            logger.AppendLine($"Finished all {finishedCount}. ");
            Console.WriteLine($"Finished all {finishedCount}. ");
            Output();
            hasFinished = true;
        }

        private static void TryGetBaseAlphas(string filename)
        {
            try
            {
                List<Instance> instances = CSV.ReadFromCsv(filename, null);
                StringBuilder sb = new StringBuilder("feature0,feature1,label,alpha\r\n");
                foreach ((Instance instance, double alpha) in new KNNContext(instances).GetAllAlphaValues())
                {
                    sb.AppendLine($"{instance.Serialize()},{alpha}");
                }
                using StreamWriter sw = new StreamWriter($"..\\a270 with alpha\\{Path.GetFileNameWithoutExtension(filename)}.csv");
                sw.Write(sb);
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
            using StreamWriter logWriter = new StreamWriter("..\\log.txt");
            logWriter.Write(logger);
        }
#endif
        #endregion

        #region ALPHA_TO_BINFREQ
#if ALPHA_TO_BINFREQ
        static readonly StringBuilder resultsBuilder = new StringBuilder("filename,oalpha-bin0,oalpha-bin1,oalpha-bin2,oalpha-bin3,oalpha-bin4,oalpha-bin5,oalpha-bin6,oalpha-bin7,oalpha-bin8,oalpha-bin9,knnallrew-bin0,knnallrew-bin1,knnallrew-bin2,knnallrew-bin3,knnallrew-bin4,knnallrew-bin5,knnallrew-bin6,knnallrew-bin7,knnallrew-bin8,knnallrew-bin9,nbpkid-bin0,nbpkid-bin1,nbpkid-bin2,nbpkid-bin3,nbpkid-bin4,nbpkid-bin5,nbpkid-bin6,nbpkid-bin7,nbpkid-bin8,nbpkid-bin9,dtc44-bin0,dtc44-bin1,dtc44-bin2,dtc44-bin3,dtc44-bin4,dtc44-bin5,dtc44-bin6,dtc44-bin7,dtc44-bin8,dtc44-bin9,knnallrew-adiff-bin0,knnallrew-adiff-bin1,knnallrew-adiff-bin2,knnallrew-adiff-bin3,knnallrew-adiff-bin4,knnallrew-adiff-bin5,knnallrew-adiff-bin6,knnallrew-adiff-bin7,knnallrew-adiff-bin8,knnallrew-adiff-bin9,nbpkid-adiff-bin0,nbpkid-adiff-bin1,nbpkid-adiff-bin2,nbpkid-adiff-bin3,nbpkid-adiff-bin4,nbpkid-adiff-bin5,nbpkid-adiff-bin6,nbpkid-adiff-bin7,nbpkid-adiff-bin8,nbpkid-adiff-bin9,dtc44-adiff-bin0,dtc44-adiff-bin1,dtc44-adiff-bin2,dtc44-adiff-bin3,dtc44-adiff-bin4,dtc44-adiff-bin5,dtc44-adiff-bin6,dtc44-adiff-bin7,dtc44-adiff-bin8,dtc44-adiff-bin9\r\n");
        static int finishedCount = 0;

        static void Main()
        {
            foreach (string filename in Directory.EnumerateFiles($"{Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)}\\source\\repos\\MachineLearning\\Dataset\\B739\\B739-allAlphas"))
            {
                CalcBinFreq(filename);
            }
            using StreamWriter sw = new StreamWriter("..\\B739-allBinFreqs.csv");
            sw.Write(resultsBuilder);
        }

        private static void CalcBinFreq(string filename)
        {
            StringBuilder sb = new StringBuilder(Path.GetFileNameWithoutExtension(filename));
            Table<string> table = CSV.ReadFromCsv(filename, true);

            foreach (Index index in new Index[] { 3, 7, 12, 17 })
            {
                List<double> valueColumn = table.SelectColumn(index).ConvertAll(s => double.Parse(s));
                double[] binFreq = new double[10];
                for (int i = 0; i < 10; i++)
                {
                    double binLowerBound = i / 10.0;
                    double binUpperBound = i == 9 ? 1.01 : (i + 1) / 10.0;
                    binFreq[i] = valueColumn.Count(d => d < binUpperBound && d >= binLowerBound) / (double)valueColumn.Count;
                }
                sb.Append("," + string.Join(',', binFreq));
            }

            foreach (Index index in new Index[] { 8, 13, 18 })
            {
                List<double> valueColumn = table.SelectColumn(index).ConvertAll(s => double.Parse(s));
                double[] binFreq = new double[10];
                for (int i = -5; i < 5; i++)
                {
                    double binLowerBound = i / 5.0;
                    double binUpperBound = i == 4 ? 1.01 : (i + 1) / 5.0;
                    binFreq[i + 5] = valueColumn.Count(d => d < binUpperBound && d >= binLowerBound) / (double)valueColumn.Count;
                }
                sb.Append("," + string.Join(',', binFreq));
            }

            resultsBuilder.AppendLine(sb.ToString());
            Console.WriteLine(++finishedCount);
        }
#endif
        #endregion

        #region XFCV_CMD
#if XFCV_CMD
        static void Main()
        {
            int finishedCount = 0;
            StringBuilder sb = new StringBuilder("filename,dtc44-cv0,dtc44-cv1,dtc44-cv2,dtc44-cv3,dtc44-cv4,dtc44-cv5,dtc44-cv6,dtc44-cv7,dtc44-cv8,dtc44-cv9\r\n");
            foreach (string filename in Directory.EnumerateFiles("..\\ECOC8030-arff").Select(filename => Path.GetFileNameWithoutExtension(filename)))
            {
                double[] accuracy = new double[10];
                for (int i = 0; i < 10; i++)
                {
                    string[] rows = File.ReadAllLines($"..\\ECOC8030-xfcv\\{filename}.{i}.csv")[1..^1];
                    accuracy[i] = rows.Count(row => string.IsNullOrWhiteSpace(row.Split(',')[3])) / (double)rows.Length;
                }
                sb.AppendLine($"{filename[..^1]},{string.Join(',', accuracy)}");
                Console.WriteLine(++finishedCount);
            }
            using StreamWriter sw = new StreamWriter("..\\ECOC8030-dtc44-accuracy.csv");
            sw.Write(sb);
        }

#endif
        #endregion

        #region A270_RESAMPLE
#if A270_RESAMPLE
        static void Main()
        {
            Random random = new Random();
            Directory.SetCurrentDirectory($"{Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)}\\source\\repos\\MachineLearning\\Dataset\\a270");
            foreach (string filename in Directory.EnumerateFiles(".\\original\\a270-raw by label\\3"))
            {
                for (int i = 1; i <= 2; i++)
                {
                    List<string> rows = File.ReadAllLines(filename).ToList();
                    List<string> newRows = new List<string>();
                    for (int j = 0; j < 562; j++)
                    {
                        int rowNumber = random.Next(rows.Count);
                        newRows.Add(rows[rowNumber]);
                        rows.RemoveAt(rowNumber);
                    }
                    File.WriteAllLines($".\\RES Test 3\\Test3-new\\{Path.GetFileNameWithoutExtension(filename)}.RES{i}.csv", newRows);
                }
            }
        }
#endif
        #endregion

        #region BETA_EXPR
#if BETA_EXPR
        public static int finishedCount = 0;
        public static DateTime programStartTime = DateTime.Now;
        public static TimeSpan totalProcessTime = TimeSpan.Zero;
        public static void Main() => Parallel.ForEach(Directory.EnumerateFiles("..\\pending"), new ParallelOptions { MaxDegreeOfParallelism = 2 * Environment.ProcessorCount }, filename => CalcBeta(filename));

        public static void CalcBeta(string filename)
        {
            DateTime processStartTime = DateTime.Now;

            List<Instance> instances = CSV.ReadFromCsv(filename, null);
            StringBuilder sb = new StringBuilder($"{string.Join(',', instances.First().Features.Select(f => f.Name))},label,beta\r\n");
            foreach ((Instance instance, double beta) in new KNNContext(instances).GetAllBetaValues())
            {
                sb.AppendLine($"{instance.Serialize()},{beta}");
            }
            File.WriteAllText($"..\\results\\{Path.GetFileName(filename)}", sb.ToString());
            File.Move(filename, $"..\\finished\\{Path.GetFileName(filename)}");

            DateTime processEndTime = DateTime.Now;
            TimeSpan processTimeSpan = processEndTime - processStartTime;
            Console.WriteLine($"{processEndTime}\t{++finishedCount}\t{Path.GetFileNameWithoutExtension(filename)}\t\t{processTimeSpan:hh\\:mm\\:ss}\t{totalProcessTime += processTimeSpan:hh\\:mm\\:ss}\t{processEndTime - programStartTime:hh\\:mm\\:ss}");
        }
#endif
        #endregion
    }
}
