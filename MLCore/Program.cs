using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using MLCore.Algorithm;

namespace MLCore
{
    static class Program
    {
        //static int finishedCount = 0;
        //static void Main()
        //{
        //    foreach (string filename in Directory.EnumerateFiles("..\\..\\artificial-R-all"))
        //    {
        //        TryCalcProbDist(filename);
        //    }
        //}

        //private static void TryCalcProbDist(string filename)
        //{
        //    try
        //    {
        //        StringBuilder sb = new StringBuilder($"feature0,feature1,label,dtc44-p0,dtc44-p1\r\n");
        //        List<Instance> instances = new List<Instance>();
        //        foreach (string row in File.ReadLines(filename))
        //        {
        //            if (row.StartsWith("feature0"))
        //            {
        //                continue;
        //            }
        //            string[] fields = row.Split(',');
        //            instances.Add(new Instance(new Dictionary<string, Feature>() { { "feature0", new Feature(ValueType.Continuous, double.Parse(fields[0])) }, { "feature1", new Feature(ValueType.Continuous, double.Parse(fields[1])) } }, labelValue: fields[2]));
        //        }
        //        DecisionTreeContext context = new DecisionTreeContext(instances);
        //        context.Train();
        //        foreach (Instance instance in instances)
        //        {
        //            Dictionary<string, double> probDist = context.GetProbDist(instance);
        //            sb.AppendLine(string.Join(',', instance.Features["feature0"].Value, instance.Features["feature1"].Value, instance.LabelValue, probDist.ContainsKey("0.0") ? probDist["0.0"] : 0.0, probDist.ContainsKey("1.0") ? probDist["1.0"] : 0.0));
        //        }
        //        using StreamWriter sw = new StreamWriter($"..\\..\\dtc44\\{Path.GetFileNameWithoutExtension(filename)}-dtc44.csv");
        //        sw.Write(sb);
        //        Console.WriteLine(++finishedCount);
        //    }
        //    catch (Exception e)
        //    {
        //        Console.WriteLine(new string('>', Console.WindowWidth - 1));
        //        Console.WriteLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
        //        Console.WriteLine(e.ToString());
        //        Console.WriteLine(new string('>', Console.WindowWidth - 1));
        //    }
        //}
        static readonly (Type, string) alg = (typeof(NaiveBayesContext), "nb");
        static readonly StringBuilder logger = new StringBuilder();
        static readonly StringBuilder resultsBuilder = new StringBuilder($"filename,{alg.Item2}-cv0,{alg.Item2}-cv1,{alg.Item2}-cv2,{alg.Item2}-cv3,{alg.Item2}-cv4,{alg.Item2}-cv5,{alg.Item2}-cv6,{alg.Item2}-cv7,{alg.Item2}-cv8,{alg.Item2}-cv9\r\n");
        static int finishedCount = 0;

        static void Main()
        {
            AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;
            //foreach (string filename in Directory.EnumerateFiles("..\\..\\..\\..\\Dataset\\temp59"))
            //{
            //    TryCvAccuracy(filename);
            //}
            Parallel.ForEach(Directory.EnumerateFiles("..\\..\\..\\..\\Dataset\\temp59"), new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, filename => TryCvAccuracy(filename));
            logger.AppendLine($"Finished all {finishedCount}. ");
            Console.WriteLine($"Finished all {finishedCount}. ");
            Output();
        }

        static void CurrentDomain_ProcessExit(object? sender, EventArgs e)
        {
            logger.AppendLine($"Program exited after finishing {finishedCount}. ");
            Output();
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
            //List<Instance> instances = new List<Instance>();
            //CSV.ReadFromCsv(filename, true).ToList().ForEach(row => instances.Add(new Instance(new Dictionary<string, Feature>() { { "feature0", new Feature(ValueType.Continuous, double.Parse(row[0])) }, { "feature1", new Feature(ValueType.Continuous, double.Parse(row[1])) } }, row[2])));
            List<Instance> instances = CSV.ReadFromCsv(filename, null);
            filename = Path.GetFileNameWithoutExtension(filename);
            //try
            //{
            double[] accuracyValues = new double[10];
            for (int i = 0; i < 10; i++)
            {
                CrossValidation.CvPrediction(instances, alg.Item1, out accuracyValues[i]);
            }
            resultsBuilder.AppendLine($"{filename},{string.Join(',', accuracyValues)}");
            logger.AppendLine($"{DateTime.Now}\tSuccessfully finished {filename} (Total: {++finishedCount})");
            Console.WriteLine($"{DateTime.Now}\tSuccessfully finished {filename} (Total: {finishedCount})");
            Console.WriteLine($"{filename},{string.Join(',', accuracyValues)}");
            //}
            //catch (Exception e)
            //{
            //    Console.WriteLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
            //    resultsBuilder.AppendLine($"{filename},{string.Join(',', Enumerable.Repeat("NaN", 10))}");
            //    logger.AppendLine(new string('>', 64));
            //    logger.AppendLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
            //    logger.AppendLine(e.ToString());
            //    logger.AppendLine(new string('>', 64));
            //}
        }
    }
}
