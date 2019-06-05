using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MLCore
{
    public static class Generation
    {
        /// <summary>
        /// Generates a 2-dimensional (values of both features are continuous), binary-labeled dataset, in which the decision boundaries are straight lines ("separators") originate from the center of the dataset, i.e. (0.5, 0.5). 
        /// </summary>
        /// <param name="separatorCount">The number of separators acting as decision boundaries. Should be an even number. </param>
        /// <param name="testTemplate">Instances to be tested on for the separators. </param>
        /// <param name="randomSeed">The seed used to initialize a Random instance. If left null, a parameterless constructor will be used. </param>
        /// <param name="logFilename">The filename, including extension, of the location that the angles relative to the center of the dataset of the generated separators to be saved. If left null, such info will be discarded. </param>
        /// <returns>A list of Instance representing the dataset generated. </returns>
        public static List<Instance> GenerateLinearSeperated(int separatorCount, List<Instance> testTemplate, int? randomSeed = null, string? logFilename = null)
        {
            Random random = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
            List<double> angles = new List<double>();
            for (int i = 0; i < separatorCount; i++)
            {
                angles.Add(Math.PI - random.NextDouble() * 2 * Math.PI);
            }
            angles.Add(Math.PI);
            angles.Sort();

            // Math.Atan2() returns double value x that -PI < x <= PI. 
            //                 PI / 2
            //                    |
            //       2nd quad.    |    1st quad. 
            //                    |
            // PI ----------------+---------------- 0
            //                    |
            //       3rd quad.    |    4th quad. 
            //                    |
            //                -PI / 2
            // The result after sorting represents the values, in sequence, 
            // in the 3rd, 4th, 1st, and finally the 2nd quadrants. 

            if (!(logFilename is null))
            {
                StringBuilder sb = new StringBuilder();
                angles.ForEach(d => sb.Append($"{d}, "));
                File.WriteAllText(logFilename, sb.ToString().Substring(0, sb.Length - 2));
            }

            List<Instance> result = new List<Instance>();
            foreach (Instance testingInstance in testTemplate)
            {
                double relativeAngle = Math.Atan2(testingInstance.Features["feature1"].Value - 0.5, testingInstance.Features["feature0"].Value - 0.5);
                result.Add(new Instance(testingInstance.Features, $"{GetRegionIndex(angles, relativeAngle) % 2}.0", testingInstance.LabelName));
            }
            return result;

            static int GetRegionIndex(List<double> thresholdsSorted, double testValue)
            {
                if (testValue <= thresholdsSorted[0])
                {
                    return 0;
                }
                int minIndex = 1;
                int maxIndex = thresholdsSorted.Count - 1;
                int midIndex = -1;
                while (minIndex <= maxIndex)
                {
                    midIndex = (minIndex + maxIndex) / 2;
                    if (thresholdsSorted[midIndex] < testValue)
                    {
                        minIndex = midIndex + 1;
                        continue;
                    }
                    if (testValue <= thresholdsSorted[midIndex - 1])
                    {
                        maxIndex = midIndex - 1;
                        continue;
                    }
                    break;
                }
                return midIndex;
            }
        }
    }
}
