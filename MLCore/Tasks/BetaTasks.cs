namespace MLCore.Tasks
{
    public static class BetaTasks
    {
        #region BETA_EXPR
#if BETA_EXPR
        public static int finishedCount = 0;

        public static void Main(string[] args) => Parallel.ForEach(
            Directory.EnumerateFiles(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "source\\repos\\MachineLearning\\Dataset\\B739\\B739-raw")),
            new ParallelOptions { MaxDegreeOfParallelism = (int)(args.Length == 0 ? 1.0 : double.Parse(args[0]) * Environment.ProcessorCount) },
            filename => CalcBeta(filename));

        public static void CalcBeta(string filename)
        {
            List<Instance> instances = CSV.ReadFromCsv(filename, null);
            List<string> outputLines = new List<string>() { "feature0,feature1,label,beta" };
            foreach ((Instance i, double b) in new KNNContext(instances).GetAllBetaValues())
            {
                outputLines.Add($"{i.Serialize()},{b}");
            }
            File.WriteAllLines(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), $"source\\repos\\MachineLearning\\Dataset\\B739\\B739-beta\\{Path.GetFileName(filename)}"), outputLines);
            Console.WriteLine($"{++finishedCount}\t{filename}");
        }
#endif
        #endregion

        #region BETA_TO_BINFREQ
#if BETA_TO_BINFREQ
        public static void Main()
        {
            List<string> lines = new List<string>() { "datasetName,beta-bin0,beta-bin1,beta-bin2,beta-bin3,beta-bin4,beta-bin5,beta-bin6,beta-bin7,beta-bin8,beta-bin9" };
            foreach (string filename in Directory.EnumerateFiles(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "source\\repos\\MachineLearning\\Dataset\\B739\\B739-beta")))
            {
                int[] binFreqs = new int[10];
                int rowCount = 0;
                foreach (string row in File.ReadAllLines(filename)[1..])
                {
                    decimal beta = decimal.Parse(row.Split(',')[^1]);
                    ++rowCount;
                    ++binFreqs[beta >= 1 ? 9 : (int)(beta * 10)];
                }
                lines.Add($"{Path.GetFileNameWithoutExtension(filename)},{string.Join(',', binFreqs.Select(i => i / (decimal)rowCount))}");
            }
            File.WriteAllLines(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "source\\repos\\MachineLearning\\Dataset\\B739\\B739-betaBinFreqs.csv"), lines);
        }
#endif
        #endregion

        #region DERIVED_BETA
#if DERIVED_BETA
        static bool isProgramInterrupted = false, isProgramFinished = false;
        static int finishedCount = 0;

        public static void Main(string[] args)
        {
            AssemblyName executingAssemblyName = Assembly.GetExecutingAssembly().GetName();
            Console.WriteLine($"{executingAssemblyName.Name} {executingAssemblyName.Version}");

            Console.CancelKeyPress += Console_CancelKeyPress;
            AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;

            Console.WriteLine($"Max degree of Parallelism: {Environment.ProcessorCount}");

            Parallel.ForEach(Directory.EnumerateFiles("../pending"), new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, filename => TryGetDerivedBeta(filename));
            isProgramFinished = true;
        }

        static void TryGetDerivedBeta(string filename)
        {
            if (!isProgramInterrupted)
            {
                try
                {
                    DateTime startTime = DateTime.Now;
                    Table<string> table = CSV.ReadFromCsv(filename, true);
                    filename = Path.GetFileName(filename);

                    List<string> labels = table.SelectColumn(^22);
                    List<double> knnp0 = table.SelectColumn(^15, s => double.Parse(s));
                    List<double> knnp1 = table.SelectColumn(^14, s => double.Parse(s));
                    List<double> nbp0 = table.SelectColumn(^10, s => double.Parse(s));
                    List<double> nbp1 = table.SelectColumn(^9, s => double.Parse(s));
                    List<double> dtp0 = table.SelectColumn(^5, s => double.Parse(s));
                    List<double> dtp1 = table.SelectColumn(^4, s => double.Parse(s));

                    List<Instance> knnDerivedInstances = new List<Instance>();
                    List<Instance> nbDerivedInstances = new List<Instance>();
                    List<Instance> dtDerivedInstances = new List<Instance>();

                    for (int i = 0; i < labels.Count; i++)
                    {
                        knnDerivedInstances.Add(new Instance(
                            new List<Feature>() { new Feature("knnp0", ValueType.Continuous, knnp0[i]),
                                new Feature("knnp1", ValueType.Continuous, knnp1[i]) }, labels[i]));
                        nbDerivedInstances.Add(new Instance(
                            new List<Feature>() { new Feature("nbp0", ValueType.Continuous, nbp0[i]),
                                new Feature("nbp1", ValueType.Continuous, nbp1[i]) }, labels[i]));
                        dtDerivedInstances.Add(new Instance(
                            new List<Feature>() { new Feature("dtp0", ValueType.Continuous, dtp0[i]),
                                new Feature("dtp1", ValueType.Continuous, dtp1[i]) }, labels[i]));
                    }

                    List<(Instance _, double beta)> knnDerivedBetas = new KNNContext(knnDerivedInstances).GetAllBetaValues().ToList();
                    List<(Instance _, double beta)> nbDerivedBetas = new KNNContext(nbDerivedInstances).GetAllBetaValues().ToList();
                    List<(Instance _, double beta)> dtDerivedBetas = new KNNContext(dtDerivedInstances).GetAllBetaValues().ToList();

                    List<string> lines = new List<string>() { "label,knnallrew-beta,nbpkid-beta,dtc44-beta" };
                    for (int i = 0; i < labels.Count; i++)
                    {
                        lines.Add(string.Join(',', labels[i], knnDerivedBetas[i].beta, nbDerivedBetas[i].beta, dtDerivedBetas[i].beta));
                    }
                    File.WriteAllLines($"../out/{filename}", lines);
                    File.Move($"../pending/{filename}", $"../finished/{filename}");
                    Console.WriteLine($"{++finishedCount}\t{startTime}\t{DateTime.Now}\t{filename}");
                }
                catch (Exception e)
                {
                    Console.WriteLine($"{filename}: {e.Message}");
                    Console.WriteLine(e.StackTrace);
                }
            }
        }

        static void Console_CancelKeyPress(object sender, ConsoleCancelEventArgs e)
        {
            isProgramInterrupted = true;
            Console.WriteLine("Cancel command received. Will stop processing new tasks. ");
            e.Cancel = true;
        }

        static void CurrentDomain_ProcessExit(object? sender, EventArgs e)
        {
            if (!isProgramInterrupted)
            {
                isProgramInterrupted = true;
                if (!isProgramFinished)
                {
                    Console.WriteLine("Exit command received. Will stop processing new tasks. ");
                }
                while (!isProgramFinished)
                {
                    Thread.Sleep(1000);
                }
            }
        }
#endif
        #endregion

        #region BETA_ANALYSIS
#if BETA_ANALYSIS
        public static void Main()
        {
            int finishedCount = 0;
            Directory.SetCurrentDirectory(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "ML\\beta work"));
            List<string> lines = new List<string>() { "filename,base_beta-bin0,base_beta-bin1,base_beta-bin2,base_beta-bin3,base_beta-bin4,knn_beta-bin0,knn_beta-bin1,knn_beta-bin2,knn_beta-bin3,knn_beta-bin4,nb_beta-bin0,nb_beta-bin1,nb_beta-bin2,nb_beta-bin3,nb_beta-bin4,dt_beta-bin0,dt_beta-bin1,dt_beta-bin2,dt_beta-bin3,dt_beta-bin4" };
            foreach (string filename in Directory.EnumerateFiles(".\\a270 original dt beta").Select(s => Path.GetFileNameWithoutExtension(s)))
            {
                List<decimal> baseB = new List<decimal>();
                CSV.ReadFromCsv($".\\a270 original base beta\\{filename}_beta1.csv", true).SelectColumn(^1).ForEach(s => { if (s != "NaN") baseB.Add(decimal.Parse(s)); });
                decimal instancesCount = baseB.Count;

                Table<string> knnAndNbBetas = CSV.ReadFromCsv($".\\a270 original knn and nb beta\\{filename}.csv", true);
                List<decimal> knnB = new List<decimal>();
                List<decimal> nbB = new List<decimal>();
                Table<string> dtBetas = CSV.ReadFromCsv($".\\a270 original dt beta\\{filename}.csv", true);
                List<decimal> dtB = new List<decimal>();

                knnAndNbBetas.SelectColumn(^4).ForEach(s => { if (s != "NaN") knnB.Add(decimal.Parse(s)); });
                knnAndNbBetas.SelectColumn(^1).ForEach(s => { if (s != "NaN") nbB.Add(decimal.Parse(s)); });
                dtBetas.SelectColumn(^1).ForEach(s => { if (s != "NaN") dtB.Add(decimal.Parse(s)); });

                decimal[] binFreqs = new decimal[20];
                for (int i = 0; i < 5; i++)
                {
                    decimal lowerBound = i / 5.0M;
                    decimal upperBound = i == 4 ? 1.01M : (i + 1) / 5.0M;
                    binFreqs[i] = baseB.Count(d => d >= lowerBound && d < upperBound) / instancesCount;
                    binFreqs[i + 5] = knnB.Count(d => d >= lowerBound && d < upperBound) / instancesCount;
                    binFreqs[i + 10] = nbB.Count(d => d >= lowerBound && d < upperBound) / instancesCount;
                    binFreqs[i + 15] = dtB.Count(d => d >= lowerBound && d < upperBound) / instancesCount;
                }

                lines.Add($"{Path.GetFileNameWithoutExtension(filename)},{string.Join(',', binFreqs)}");
                Console.WriteLine(++finishedCount);
            }
            File.WriteAllLines(".\\a270-beta-5binFreqs.csv", lines);
        }
#endif
        #endregion
    }
}
