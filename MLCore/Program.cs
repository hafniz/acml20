using MLCore.Tasks;

namespace MLCore
{
    static class Program
    {
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

        public static void Main() => ResultAnalysis.Main();
    }
}
