﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MLCore
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
        static void Main()
        {
            List<ResultEntry> uciEntries = File.ReadAllLines(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Desktop\\ML\\202005\\predicted-labels-uci.csv"))[1..].Select(row => new ResultEntry(row)).ToList();
            List<ResultEntry> artificialEntries = File.ReadAllLines(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Desktop\\ML\\202005\\predicted-labels-artificial.csv"))[1..].Select(row => new ResultEntry(row)).ToList();
            MetaFeatureSet[] metaFeatureSets = (MetaFeatureSet[])Enum.GetValues(typeof(MetaFeatureSet));


        }

        private static void ModelConfusionMatrix(List<ResultEntry> entries, string datasetGroup, MetaFeatureSet model)
        {
            List<ResultEntry> modelEntries = entries.Where(e => e.MetaFeatureSet == model).ToList();
            List<string> lines = new List<string>() { "cvNumber,cell0,cell1,cell2,cell3,cell4,cell5,cell6,cell10,cell11,cell12,cell13,cell14,cell15,cell16,cell20,cell21,cell22,cell23,cell24,cell25,cell26" };
            for (byte cvNumber = 0; cvNumber < 10; cvNumber++)
            {
                List<ResultEntry> cvEntries = modelEntries.Where(e => e.CvNumber == cvNumber).ToList();
                int[][] cellValues = new int[3][] { new int[7], new int[7], new int[7] };
                foreach (ResultEntry entry in cvEntries)
                {
                    switch (entry.Correctness)
                    {
                        case Correctness.Correct:
                            ++cellValues[0][entry.ActualLabel];
                            break;
                        case Correctness.Partial:
                            ++cellValues[1][entry.ActualLabel];
                            break;
                        default:
                            ++cellValues[2][entry.ActualLabel];
                            break;
                    }
                }
                lines.Add($"{cvNumber},{string.Join(',', cellValues[0])},{string.Join(',', cellValues[1])},{string.Join(',', cellValues[2])}");
            }
            File.WriteAllLines(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), $"Desktop\\ML\\202005\\{datasetGroup}-confusionMatrix-{model}.csv"), lines);
        }

        private static void ModelComparisonMatrix(List<ResultEntry> entries, string datasetGroup, MetaFeatureSet rowModel, MetaFeatureSet columnModel)
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

                    switch (rowCorrectness)
                    {
                        case Correctness.Correct when columnCorrectness == Correctness.Correct:
                            ++cellValues[0];
                            break;
                        case Correctness.Correct when columnCorrectness == Correctness.Partial:
                            ++cellValues[1];
                            break;
                        case Correctness.Correct when columnCorrectness == Correctness.Incorrect:
                            ++cellValues[2];
                            break;
                        case Correctness.Partial when columnCorrectness == Correctness.Correct:
                            ++cellValues[3];
                            break;
                        case Correctness.Partial when columnCorrectness == Correctness.Partial:
                            ++cellValues[4];
                            break;
                        case Correctness.Partial when columnCorrectness == Correctness.Incorrect:
                            ++cellValues[5];
                            break;
                        case Correctness.Incorrect when columnCorrectness == Correctness.Correct:
                            ++cellValues[6];
                            break;
                        case Correctness.Incorrect when columnCorrectness == Correctness.Partial:
                            ++cellValues[7];
                            break;
                        default:
                            ++cellValues[8];
                            break;
                    }
                }
                string line = $"{cvNumber},{string.Join(',', cellValues)}";
                Console.WriteLine(line);
                lines.Add(line);
            }
            File.WriteAllLines(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), $"Desktop\\ML\\202005\\ModelComparisonMatrices\\{datasetGroup}-{rowModel}-{columnModel}.csv"), lines);
        }

        private static void ConsolidateData()
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
            foreach (string filename in Directory.EnumerateFiles(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Desktop\\ML\\202005\\predicted_labels_artificial")))
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
            File.WriteAllLines(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Desktop\\ML\\202005\\predicted-labels-artificial.csv"), lines);
        }
    }
}
