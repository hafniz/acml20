#define OALPHA_BINFREQ
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
                    StringBuilder sb = new StringBuilder($"{string.Join(',', instances.First().Features.Select(kvp => kvp.Key))},label,alpha\r\n");
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

        static void Main()
        {
            AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;
            //foreach (string filename in Directory.EnumerateFiles("..\\..\\..\\..\\Dataset\\artificial-R\\dataset2824 with alpha"))
            //{
            //    TryCvAccuracy(filename);
            //}
            Parallel.ForEach(Directory.EnumerateFiles("..\\ar2824"), new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, filename => TryCvAccuracy(filename));
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
            List<Instance> instances = CSV.ReadFromCsv(filename, headerNameList: "feature0,feature1,label", trimRight: 1);
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
    }
}
