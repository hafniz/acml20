﻿using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;

namespace MLCore.Tasks
{
    public enum MetaFeatureSet
    {
        Conventional = 0,
        Beta = 1,
        Both = 2
    }

    public enum Correctness
    {
        Correct = 0,
        Partial = 1,
        Incorrect = 2
    }

    public struct ResultEntry
    {
        public string DatasetName { get; set; }
        public MetaFeatureSet MetaFeatureSet { get; set; }
        public byte CvNumber { get; set; }
        public byte FoldNumber { get; set; }
        public byte ActualLabel { get; set; }
        public byte PredictedLabel { get; set; }
        public Correctness Correctness { get; set; }

        public ResultEntry(string raw)
        {
            string[] fields = raw.Split(',');
            DatasetName = fields[0];
            MetaFeatureSet = (MetaFeatureSet)Enum.Parse(typeof(MetaFeatureSet), fields[1]);
            CvNumber = byte.Parse(fields[2]);
            FoldNumber = byte.Parse(fields[3]);
            ActualLabel = byte.Parse(fields[4]);
            PredictedLabel = byte.Parse(fields[5]);
            Correctness = (Correctness)Enum.Parse(typeof(Correctness), fields[6]);
        }
        public override string ToString() => $"{DatasetName},{MetaFeatureSet},{CvNumber},{FoldNumber},{ActualLabel},{PredictedLabel},{Correctness}";
    }

    public static class ResultAnalysis
    {
        public static decimal GetModelImprovementScore(IEnumerable<ResultEntry> entries, MetaFeatureSet thisModel, MetaFeatureSet otherModel, params byte[] excludeLabels)
        {
            int cvNumberCount = entries.Select(e => e.CvNumber).Distinct().Count();
            List<string> datasetNames = entries.Where(e => !excludeLabels.Contains(e.ActualLabel)).Select(e => e.DatasetName).Distinct().ToList();
            decimal[] scores = new decimal[cvNumberCount];
            Dictionary<string, ResultEntry>[] thisModelEntries = new Dictionary<string, ResultEntry>[cvNumberCount];
            Dictionary<string, ResultEntry>[] otherModelEntries = new Dictionary<string, ResultEntry>[cvNumberCount];
            for (int cvNumber = 0; cvNumber < cvNumberCount; cvNumber++)
            {
                thisModelEntries[cvNumber] = new Dictionary<string, ResultEntry>();
                otherModelEntries[cvNumber] = new Dictionary<string, ResultEntry>();
            }
            foreach (ResultEntry entry in entries)
            {
                if (excludeLabels.Contains(entry.ActualLabel))
                {
                    continue;
                }
                if (entry.MetaFeatureSet == thisModel)
                {
                    thisModelEntries[entry.CvNumber].Add(entry.DatasetName, entry);
                }
                else if (entry.MetaFeatureSet == otherModel)
                {
                    otherModelEntries[entry.CvNumber].Add(entry.DatasetName, entry);
                }
            }
            for (int cvNumber = 0; cvNumber < cvNumberCount; cvNumber++)
            {
                foreach (string datasetName in datasetNames)
                {
                    Correctness thisModelCorrectness = thisModelEntries[cvNumber][datasetName].Correctness;
                    Correctness otherModelCorrectness = otherModelEntries[cvNumber][datasetName].Correctness;
                    scores[cvNumber] += thisModelCorrectness switch
                    {
                        Correctness.Correct when otherModelCorrectness == Correctness.Incorrect => 1,
                        Correctness.Correct when otherModelCorrectness == Correctness.Partial => 0.5M,
                        Correctness.Partial when otherModelCorrectness == Correctness.Incorrect => 0.5M,
                        Correctness.Partial when otherModelCorrectness == Correctness.Correct => -0.5M,
                        Correctness.Incorrect when otherModelCorrectness == Correctness.Partial => -0.5M,
                        Correctness.Incorrect when otherModelCorrectness == Correctness.Correct => -1,
                        _ => 0
                    };
                }
            }
            return scores.Select(s => s / datasetNames.Count()).Average();
        }

        public static void ModelConfusionMatrix(IEnumerable<ResultEntry> entries, MetaFeatureSet model, string outputFilename)
        {
            List<ResultEntry> modelEntries = entries.Where(e => e.MetaFeatureSet == model).ToList();
            List<string> lines = new List<string>() { "cvNumber,cell0,cell1,cell2,cell3,cell4,cell5,cell6,cell10,cell11,cell12,cell13,cell14,cell15,cell16,cell20,cell21,cell22,cell23,cell24,cell25,cell26" };
            for (byte cvNumber = 0; cvNumber < 10; cvNumber++)
            {
                List<ResultEntry> cvEntries = modelEntries.Where(e => e.CvNumber == cvNumber).ToList();
                int[][] cellValues = new int[3][] { new int[7], new int[7], new int[7] };
                foreach (ResultEntry entry in cvEntries)
                {
                    ++cellValues[(int)entry.Correctness][entry.ActualLabel];
                }
                lines.Add($"{cvNumber},{string.Join(',', cellValues[0])},{string.Join(',', cellValues[1])},{string.Join(',', cellValues[2])}");
            }
            File.WriteAllLines(outputFilename, lines);
        }

        public static void ModelComparisonMatrix(IEnumerable<ResultEntry> entries, MetaFeatureSet rowModel, MetaFeatureSet columnModel, string outputFilename)
        {
            List<string> datasetNames = entries.Select(e => e.DatasetName).Distinct().ToList();
            List<string> lines = new List<string>() { "cvNumber,cell1,cell2,cell3,cell4,cell5,cell6,cell7,cell8,cell9" };
            for (byte cvNumber = 0; cvNumber < 10; cvNumber++)
            {
                List<ResultEntry> cvEntries = entries.Where(e => e.CvNumber == cvNumber).ToList();
                int[] cellValues = new int[9];
                foreach (string s in datasetNames)
                {
                    Correctness rowCorrectness = cvEntries.Single(e => e.DatasetName == s && e.MetaFeatureSet == rowModel).Correctness;
                    Correctness columnCorrectness = cvEntries.Single(e => e.DatasetName == s && e.MetaFeatureSet == columnModel).Correctness;
                    ++cellValues[rowCorrectness switch
                    {
                        Correctness.Correct when columnCorrectness == Correctness.Correct => 0,
                        Correctness.Correct when columnCorrectness == Correctness.Partial => 1,
                        Correctness.Correct when columnCorrectness == Correctness.Incorrect => 2,
                        Correctness.Partial when columnCorrectness == Correctness.Correct => 3,
                        Correctness.Partial when columnCorrectness == Correctness.Partial => 4,
                        Correctness.Partial when columnCorrectness == Correctness.Incorrect => 5,
                        Correctness.Incorrect when columnCorrectness == Correctness.Correct => 6,
                        Correctness.Incorrect when columnCorrectness == Correctness.Partial => 7,
                        _ => 8
                    }];
                }
                string line = $"{cvNumber},{string.Join(',', cellValues)}";
                Console.WriteLine(line);
                lines.Add(line);
            }
            File.WriteAllLines(outputFilename, lines);
        }

        public static void WriteResultEntries(string sourceFolder, string outputFilename)
        {
            byte[,] correctnesses =
            {
                { 0, 1, 1, 1, 1, 1, 1 },
                { 2, 0, 2, 2, 2, 2, 2 },
                { 2, 2, 0, 2, 2, 2, 2 },
                { 2, 2, 2, 0, 2, 2, 2 },
                { 2, 1, 1, 2, 0, 2, 2 },
                { 2, 1, 2, 1, 2, 0, 2 },
                { 2, 2, 1, 1, 2, 2, 0 }
            };

            List<ResultEntry> entries = new List<ResultEntry>();
            foreach (string filename in Directory.EnumerateFiles(sourceFolder))
            {
                string basename = Path.GetFileNameWithoutExtension(filename);
                MetaFeatureSet metaFeatureSet = (MetaFeatureSet)byte.Parse(basename[0].ToString());
                byte cvNumber = byte.Parse(basename[2].ToString());
                byte foldNumber = byte.Parse(basename[4].ToString());
                foreach (string line in File.ReadAllLines(filename).Where(s => !string.IsNullOrWhiteSpace(s)))
                {
                    string[] fields = line.Split(',');
                    string datasetName = fields[0];
                    byte predictedLabel = byte.Parse(fields[1]);
                    byte actualLabel = byte.Parse(fields[2]);
                    Correctness correctness = (Correctness)correctnesses[actualLabel, predictedLabel];
                    entries.Add(new ResultEntry
                    {
                        MetaFeatureSet = metaFeatureSet,
                        CvNumber = cvNumber,
                        FoldNumber = foldNumber,
                        DatasetName = datasetName,
                        PredictedLabel = predictedLabel,
                        ActualLabel = actualLabel,
                        Correctness = correctness
                    });
                }
            }
            List<string> lines = new List<string>() { "DatasetName,MetaFeatureSet,CvNumber,FoldNumber,ActualLabel,PredictedLabel,Correctness" };
            lines.AddRange(entries.Select(entry => entry.ToString()));
            File.WriteAllLines(outputFilename, lines);
        }

        public static void WriteAverageAccuracy(string sourceFilename, string outputFilename)
        {
            List<string> outputLines = new List<string>() { "datasetName,knnallrew,nbpkid,dtc44" };
            foreach (string row in File.ReadAllLines(sourceFilename)[1..])
            {
                string[] fields = row.Split(',');
                string datasetName = fields[0];
                decimal knnallrew = fields[1..11].Average(s => decimal.Parse(s));
                decimal nbpkid = fields[11..21].Average(s => decimal.Parse(s));
                decimal dtc44 = fields[21..].Average(s => decimal.Parse(s));
                outputLines.Add($"{datasetName},{knnallrew},{nbpkid},{dtc44}");
            }
            File.WriteAllLines(outputFilename, outputLines);
        }

        #region ScatterPlotSource
        public static void CategorizeEntriesInConfusionMatrix(IEnumerable<ResultEntry> entries, string avgAccuracyFilename, string outputFilename, MetaFeatureSet model)
        {
            Dictionary<string, string> avgAccuracies = new Dictionary<string, string>();
            foreach (string row in File.ReadAllLines(avgAccuracyFilename)[1..])
            {
                avgAccuracies.Add(row.Split(',')[0], row);
            }

            List<string> outputLines = new List<string>() { "datasetName,knnallrew,nbpkid,dtc44,correct.Freq,partial.Freq,incorrect.Freq,category" };
            foreach (string datasetName in entries.Select(e => e.DatasetName).Distinct())
            {
                StringBuilder lineBuilder = new StringBuilder(avgAccuracies[datasetName]);
                int[] catFreqs = new int[3]; // [0]: correct, [1]: partial, [2]: incorrect
                foreach (ResultEntry entry in entries.Where(e => e.DatasetName == datasetName && e.MetaFeatureSet == model))
                {
                    ++catFreqs[(int)entry.Correctness];
                }
                lineBuilder.Append(',' + string.Join(',', catFreqs));
                if (catFreqs[0] > catFreqs[1] && catFreqs[0] > catFreqs[2])
                {
                    lineBuilder.Append($",C");
                }
                else if (catFreqs[1] > catFreqs[0] && catFreqs[1] > catFreqs[2])
                {
                    lineBuilder.Append($",P");
                }
                else if (catFreqs[2] > catFreqs[0] && catFreqs[2] > catFreqs[1])
                {
                    lineBuilder.Append($",I");
                }
                else
                {
                    lineBuilder.Append($",?");
                }
                outputLines.Add(lineBuilder.ToString());
            }
            File.WriteAllLines(outputFilename, outputLines);
        }

        public static void CategorizeEntriesInComparionMatrix(IEnumerable<ResultEntry> entries, string avgAccuracyFilename, string outputFilename, MetaFeatureSet rowModel, MetaFeatureSet columnModel)
        {
            Dictionary<string, string> avgAccuracies = new Dictionary<string, string>();
            foreach (string row in File.ReadAllLines(avgAccuracyFilename)[1..])
            {
                avgAccuracies.Add(row.Split(',')[0], row);
            }

            List<string> outputLines = new List<string>() { "datasetName,knnallrew,nbpkid,dtc44,draw.Freq,rowModelBetter.Freq,columnModelBetter.Freq,category" };
            foreach (string datasetName in entries.Select(e => e.DatasetName).Distinct())
            {
                StringBuilder lineBuilder = new StringBuilder(avgAccuracies[datasetName]);
                int[] catFreqs = new int[3]; // [0]: draw, [1]: rowModelBetter, [2]: columnModelBetter
                List<ResultEntry> datasetEntries = entries.Where(e => e.DatasetName == datasetName).ToList();
                for (int cvNumber = 0; cvNumber < 10; cvNumber++)
                {
                    Correctness rowCorrectness = datasetEntries.Single(e => e.CvNumber == cvNumber && e.MetaFeatureSet == rowModel).Correctness;
                    Correctness columnCorrectness = datasetEntries.Single(e => e.CvNumber == cvNumber && e.MetaFeatureSet == columnModel).Correctness;
                    if (rowCorrectness == columnCorrectness)
                    {
                        ++catFreqs[0];
                    }
                    else if (rowCorrectness == Correctness.Correct || rowCorrectness == Correctness.Partial && columnCorrectness == Correctness.Incorrect)
                    {
                        ++catFreqs[1];
                    }
                    else
                    {
                        ++catFreqs[2];
                    }
                }
                lineBuilder.Append(',' + string.Join(',', catFreqs));
                if (catFreqs[0] > catFreqs[1] && catFreqs[0] > catFreqs[2])
                {
                    lineBuilder.Append($",D");
                }
                else if (catFreqs[1] > catFreqs[0] && catFreqs[1] > catFreqs[2])
                {
                    lineBuilder.Append($",R");
                }
                else if (catFreqs[2] > catFreqs[0] && catFreqs[2] > catFreqs[1])
                {
                    lineBuilder.Append($",C");
                }
                else
                {
                    lineBuilder.Append($",?");
                }
                outputLines.Add(lineBuilder.ToString());
            }
            File.WriteAllLines(outputFilename, outputLines);
        }

        public static void WriteScatterPlotSourceWithCustomRgb(string accuracyFilename, string labelFilename, string outputFilename)
        {
            List<string> outputLines = new List<string>() { "datasetName,knnallrew,nbpkid,dtc44,r,g,b" };
            Dictionary<string, string> positions = new Dictionary<string, string>();
            foreach (string line in File.ReadAllLines(accuracyFilename)[1..])
            {
                positions.Add(line.Split(',')[0], line);
            }
            foreach (string line in File.ReadAllLines(labelFilename)[1..])
            {
                string[] fields = line.Split(',');
                string color = fields[1] switch
                {
                    "0" => "255,255,255",
                    "1" => "0,0,255",
                    "2" => "255,0,0",
                    "3" => "0,255,0",
                    "4" => "255,0,255",
                    "5" => "0,255,255",
                    "6" => "255,255,0",
                    _ => throw new NotImplementedException()
                };
                outputLines.Add($"{positions[fields[0]]},{color}");
            }
            File.WriteAllLines(outputFilename, outputLines);
        }
        #endregion
    }
}
