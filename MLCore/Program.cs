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
        static readonly StringBuilder sb = new StringBuilder();
        static int finishedCount = 0;
        static void Main()
        {
            IEnumerable<string> filenames = Directory.EnumerateFiles(Path.Combine(Directory.GetParent(Directory.GetCurrentDirectory()).FullName, "in"));
            Parallel.ForEach(filenames, f => TryCalcProbDist(f));
            sb.AppendLine($"Finished all {finishedCount}. ");
            Console.WriteLine($"Finished all {finishedCount}. ");
            WriteLog();
        }

        static void TryCalcProbDist(string filename)
        {
            try
            {
                CalcProbDist(filename);
            }
            catch (Exception e)
            {
                Console.WriteLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
                sb.AppendLine(new string('>', 64));
                sb.AppendLine($"{DateTime.Now}\t{e.GetType().ToString()} encountered in processing {filename}, skipping this file");
                sb.AppendLine($"{DateTime.Now}\t{e.ToString()}");
                sb.AppendLine(new string('>', 64));
            }
        }

        static void CalcProbDist(string filename)
        {
            // Prepare instances
            Table<string> fields = CSV.ReadFromCsv(filename, true);
            List<Instance> instances = new List<Instance>();
            Dictionary<Instance, Dictionary<string, double>> output = new Dictionary<Instance, Dictionary<string, double>>();
            foreach (List<string> row in fields)
            {
                Instance instance = new Instance(new Dictionary<string, Feature>() { { "feature0", new Feature(ValueType.Continuous, double.Parse(row[0])) }, { "feature1", new Feature(ValueType.Continuous, double.Parse(row[1])) } }, row[2]);
                instances.Add(instance);
                output.Add(instance, new Dictionary<string, double>());
                output[instance].Add("feature0", double.Parse(row[0]));
                output[instance].Add("feature1", double.Parse(row[1]));
                output[instance].Add("label", double.Parse(row[2]));
                output[instance].Add("alpha", double.Parse(row[3]));
            }

            // Do cross validation
            //foreach (KeyValuePair<Type, string> alg in new Dictionary<Type, string>() { { typeof(KNNContext), "ann" }, { typeof(NaiveBayesContext), "nb" }, { typeof(DecisionTreeContext), "dt" } })
            foreach (KeyValuePair<Type, string> alg in new Dictionary<Type, string>() { { typeof(KNNContext), "knn" } })
            {
                for (int cvNumber = 0; cvNumber < 10; cvNumber++)
                {
                    Dictionary<Instance, (Dictionary<string, double>, int)> cvResults = CrossValidation.CVProbDist(instances, alg.Key);
                    foreach (KeyValuePair<Instance, (Dictionary<string, double> probDist, int foldNumber)> instanceResults in cvResults)
                    {
                        output[instanceResults.Key].Add($"{alg.Value}-cv{cvNumber}-fold", instanceResults.Value.foldNumber);
                        output[instanceResults.Key].Add($"{alg.Value}-cv{cvNumber}-p0", instanceResults.Value.probDist.ContainsKey("0.0") ? instanceResults.Value.probDist["0.0"] : 0);
                        output[instanceResults.Key].Add($"{alg.Value}-cv{cvNumber}-p1", instanceResults.Value.probDist.ContainsKey("1.0") ? instanceResults.Value.probDist["1.0"] : 0);
                        output[instanceResults.Key].Add($"{alg.Value}-cv{cvNumber}-prediction", output[instanceResults.Key][$"{alg.Value}-cv{cvNumber}-p0"] > output[instanceResults.Key][$"{alg.Value}-cv{cvNumber}-p1"] ? 0.0 : 1.0);
                    }
                }
            }

            // Output result
            IEnumerable<string> headers = output.First().Value.Select(kvp => kvp.Key);
            List<List<string>> table = new List<List<string>>();
            foreach (KeyValuePair<Instance, Dictionary<string, double>> kvp in output)
            {
                List<string> row = new List<string>();
                foreach (string header in headers)
                {
                    row.Add(kvp.Value[header].ToString());
                }
                table.Add(row);
            }
            CSV.WriteToCsv(Path.Combine(Directory.GetParent(Directory.GetCurrentDirectory()).FullName, "out", Path.GetFileName(filename)), new Table<string>(table), string.Join(',', headers));
            sb.AppendLine($"{DateTime.Now}\tSuccessfully finished {filename} (Total: {++finishedCount})");
            Console.WriteLine($"{DateTime.Now}\tSuccessfully finished {filename} (Total: {finishedCount})");
        }

        static void WriteLog()
        {
            using StreamWriter sw = new StreamWriter(Path.Combine(Directory.GetParent(Directory.GetCurrentDirectory()).FullName, "log.txt"));
            sw.Write(sb);
        }
    }
}
