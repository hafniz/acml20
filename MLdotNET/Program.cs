using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace myMLApp
{
    class Program
    {
        public class DataEntry
        {
            [LoadColumn(0)] public float Feature1;
            [LoadColumn(1)] public float Feature2;
            [LoadColumn(2)] public float Feature3;
            [LoadColumn(3)] public float Feature4;
            [LoadColumn(4)] public string Label;
        }

        public class PredictionEntry
        {
            [ColumnName("PredictedLabel")] public string PredictedLabel;
        }

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<DataEntry>("iris-data.txt", ',');

            EstimatorChain<KeyToValueMappingTransformer> pipeline = mlContext.Transforms.Conversion.MapValueToKey(nameof(DataEntry.Label))
                .Append(mlContext.Transforms.Concatenate("Features", nameof(DataEntry.Feature1), nameof(DataEntry.Feature2), nameof(DataEntry.Feature3), nameof(DataEntry.Feature4)))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(nameof(PredictionEntry.PredictedLabel)));

            TransformerChain<KeyToValueMappingTransformer> model = pipeline.Fit(trainingDataView);

            PredictionEntry prediction = mlContext.Model.CreatePredictionEngine<DataEntry, PredictionEntry>(model).Predict(
                new DataEntry()
                {
                    Feature1 = float.Parse(Console.ReadLine()),
                    Feature2 = float.Parse(Console.ReadLine()),
                    Feature3 = float.Parse(Console.ReadLine()),
                    Feature4 = float.Parse(Console.ReadLine()),
                });

            Console.WriteLine($"PredictedLabel: {prediction.PredictedLabel}");
            Console.ReadLine();
        }
    }
}
