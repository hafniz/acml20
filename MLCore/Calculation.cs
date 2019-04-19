using System.Collections.Generic;
using System.Linq;
using MLCore.FrameworkElement;
using static System.Math;

namespace MLCore
{
    public static class Calculation
    {
        public static double EuclidianDistance(Instance instance1, Instance instance2)
        {
            double distSum = 0.0;
            for (int i = 0; i < instance1.Features.Count; i++)
            {
                distSum += Sqrt(Abs(instance1.Features[i].Value - instance2.Features[i].Value));
            }
            return distSum;
        }

        public static double Mean(List<double> numbers) => numbers.Average();
        public static double StdDev(List<double> numbers)
        {
            double mean = Mean(numbers);
            double sumOfSquaresOfDifferences = numbers.Select(v => Pow(v - mean, 2)).Sum();
            return Sqrt(sumOfSquaresOfDifferences / numbers.Count);
        }

        /// <summary>
        /// Calculates the possibility of the specified value being inside the normal distribution with specified mean value and standard deviation.
        /// </summary>
        /// <param name="value">Value to test. </param>
        /// <param name="mean">Mean value of the distribution. </param>
        /// <param name="stdDev">Standard deviation of the distribution. </param>
        /// <returns>The probability of the value being inside the distribution. </returns>
        public static double GaussianDensity(double value, double mean, double stdDev)
        {
            if (stdDev == 0)
            {
                return value == mean ? 1.0 : 0.0;
            }
            double exponent = -0.5 * Pow((value - mean) / stdDev, 2);
            return 1 / (stdDev * Sqrt(2 * PI)) * Pow(E, exponent);
        }
    }
}
