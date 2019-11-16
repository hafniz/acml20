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
            Task[] tasks = new Task[8];
            for (int i = 0; i < 8; i++)
            {
                List<string> filenames = Directory.EnumerateFiles($"..\\..\\..\\..\\Dataset\\artificial-new\\level0\\rebalanced\\R{i / 2 + 11}").ToList();
                List<string> sublist = filenames.GetRange(i % 2 == 0 ? 0 : 353, 353);
                tasks[i] = Task.Run(() => sublist.ForEach(s => CalculateAlpha(s)));
            }
            Task.WaitAll(tasks);
            Console.WriteLine("Finished all. ");

            static void CalculateAlpha(string filename)
            {
                List<Instance> instances = CSV.ReadFromCsv(filename, null);
                Dictionary<Instance, string> resultsSerialized = new Dictionary<Instance, string>();
                List<(Instance, double)> alphas = new KNNContext(instances).GetAllAlphaValues();
                foreach ((Instance instance, double alphaValue) in alphas)
                {
                    resultsSerialized.Add(instance, alphaValue.ToString());
                }
                List<List<string>> dataToWrite = new List<List<string>>();
                foreach (KeyValuePair<Instance, string> kvp in resultsSerialized)
                {
                    List<string> row = kvp.Key.Serialize().Split(',').ToList();
                    dataToWrite.Add(row.Concat(new List<string>() { kvp.Value }).ToList());
                }
                CSV.WriteToCsv(filename, new Table<string>(dataToWrite), "feature0,feature1,label,alpha");
                Console.WriteLine($"Finished {filename} ({++finishedCount} / 2824)");
            }
        }
    }
}
