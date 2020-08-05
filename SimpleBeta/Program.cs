using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SimpleBeta
{
    class Program
    {
        static string outputPath = null;
        static TimeSpan totalTime = TimeSpan.Zero;
        static readonly string resultPath = $".\\SimpleBeta-{DateTime.Now:yyyyMMdd-HHmmss}-results.csv";

        static void Main(string[] args)
        {
            List<string> argsList = args.ToList();

            if (!argsList.Contains("-i") || argsList.Count < argsList.IndexOf("-i") + 2)
            {
                Console.WriteLine("\nUsage: SimpleBeta -i source-folder [-o output-folder]\n");
                Console.WriteLine("source-folder\tThe path of the folder containing datasets to be processed. ");
                Console.WriteLine("output-folder\tThe path of the folder to which computed beta values will be written. \n");
                return;
            }

            if (argsList.Contains("-o") && argsList.Count > argsList.IndexOf("-o") + 1)
            {
                outputPath = argsList[argsList.IndexOf("-o") + 1];
            }

            File.WriteAllText(resultPath, "filename,instanceCount,featureCount,theoreticalTime,actualTime,totalTime\n");

            foreach (string filename in Directory.EnumerateFiles(argsList[argsList.IndexOf("-i") + 1]))
            {
                GetBeta(filename);
            }
        }

        static void GetBeta(string filename)
        {
            // Read in the dataset.
            string[] lines = File.ReadAllLines(filename);
            int instanceCount = lines.Length;
            int featureCount = lines[0].Split(',').Length - 1;

            double[,] featureValues = new double[instanceCount, featureCount];
            string[] labelValues = new string[instanceCount];
            double[] betaValues = new double[instanceCount];

            for (int i = 0; i < instanceCount; i++)
            {
                string[] fields = lines[i].Split(',');
                for (int j = 0; j < featureCount; j++)
                {
                    featureValues[i, j] = double.Parse(fields[j]);
                }
                labelValues[i] = fields[^1];
            }

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            // Calculate number of homo-label instances for each label
            Dictionary<string, int> homoCount = new Dictionary<string, int>(10);
            for (int i = 0; i < instanceCount; i++)
            {
                string labelValue = labelValues[i];
                if (homoCount.ContainsKey(labelValue))
                {
                    ++homoCount[labelValue];
                }
                else
                {
                    homoCount.Add(labelValue, 1);
                }
            }

            // Calculate the inverse distance between all pairs of instances. The invd between one instance and itself remains default(double), which is zero.
            double[,] invdStats = new double[instanceCount, instanceCount];
            for (int i = 1; i < instanceCount; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    invdStats[i, j] = invdStats[j, i] = Invd(i, j);
                }
            }

            // Compute beta for each instance.
            for (int i = 0; i < instanceCount; i++)
            {
                // Sum up invd to get denominator.
                double denominator = 0;
                for (int j = 0; j < instanceCount; j++)
                {
                    denominator += invdStats[i, j];
                }

                // Sort instances by proximity (invd descending).
                int[] instanceIndexesByProximity = Enumerable.Range(0, instanceCount).ToArray();
                Array.Sort(instanceIndexesByProximity, (j1, j2) =>
                {
                    double difference = invdStats[i, j1] - invdStats[i, j2];
                    return difference > 0 ? -1 : (difference == 0 ? 0 : 1);
                });

                // Sum up invd for instances with the same label among the first k = homoCount - 1 instances to get nominator.
                double nominator = 0;
                string labelValue = labelValues[i];
                for (int j = 0; j < homoCount[labelValue] - 1; j++)
                {
                    int otherInstanceIndex = instanceIndexesByProximity[j];
                    if (labelValues[otherInstanceIndex] == labelValue)
                    {
                        nominator += invdStats[i, otherInstanceIndex];
                    }
                }

                betaValues[i] = nominator / denominator;
            }

            stopwatch.Stop();
            totalTime += stopwatch.Elapsed;

            // Theoretical time complexity: O(n + n^2 m + Sum_{corresponding k for each label value}{k(n + n log n + k)}).
            double theoreticalTime = instanceCount + Math.Pow(instanceCount, 2) * featureCount + Enumerable.Sum(homoCount, kvp => kvp.Value * (instanceCount + instanceCount * Math.Log2(instanceCount) + kvp.Value));

            // Output.
            string[] outputInfo = new[] { Path.GetFileNameWithoutExtension(filename), instanceCount.ToString(), featureCount.ToString(), theoreticalTime.ToString(), stopwatch.Elapsed.ToString(), totalTime.ToString() };
            Console.WriteLine(string.Join('\t', outputInfo));
            using StreamWriter sw = File.AppendText(resultPath);
            sw.WriteLine(string.Join(',', outputInfo));

            if (outputPath != null)
            {
                File.WriteAllText(Path.Combine(outputPath, $"{Path.GetFileNameWithoutExtension(filename)}-beta.csv"), string.Join('\n', betaValues));
            }

            double Invd(int instanceIndex1, int instanceIndex2)
            {
                double squareSum = 0;
                for (int i = 0; i < featureCount; i++)
                {
                    squareSum += Math.Pow(featureValues[instanceIndex1, i] - featureValues[instanceIndex2, i], 2);
                }
                return 1 / (1 + Math.Sqrt(squareSum));
            }
        }
    }
}
