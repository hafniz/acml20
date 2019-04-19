using System;
using Microsoft.ML;

namespace MLCore
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            Console.WriteLine(mlContext);
        }
    }
}
