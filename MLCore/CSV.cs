using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace MLCore
{
    [DebuggerStepThrough]
    public static class CSV
    {
        /// <summary>
        /// Parse a CSV file and return a list of instances. 
        /// </summary>
        /// <param name="filename">Path and filename of the CSV file to be parsed. </param>
        /// <param name="featureTypes">The ValueType of each feature: 'c' for ValueType.Continuous, 'd' for ValueType.Discrete. Leaving as null marks all feature as continuous. </param>
        /// <param name="hasHeader">Whether the CSV file has a header. If true, the method will try to parse the header as names of features and label; otherwise, or when failed, will check <paramref name="headerNameList"/>. </param>
        /// <param name="headerNameList">Names of features and label, separated by commas. If not provided, the method will generate default names for features and label. </param>
        /// <returns>A list of instances parsed from the CSV file. </returns>
        public static List<Instance> ReadFromCsv(string filename, string? featureTypes = null, bool hasHeader = false, string? headerNameList = null) => ReadFromCsv(filename, .., featureTypes, hasHeader, headerNameList);

        /// <summary>
        /// Parse a range of columns of a CSV file and return a list of instances. 
        /// </summary>
        /// <param name="filename">Path and filename of the CSV file to be parsed. </param>
        /// <param name="columns">Range of columns in the CSV file to be parsed. </param>
        /// <param name="featureTypes">The ValueType of each feature: 'c' for ValueType.Continuous, 'd' for ValueType.Discrete. Leaving as null marks all feature as continuous. </param>
        /// <param name="hasHeader">Whether the CSV file has a header. If true, the method will try to parse the header as names of features and label; otherwise, or when failed, will check <paramref name="headerNameList"/>. </param>
        /// <param name="headerNameList">Names of features and label, separated by commas. If not provided, the method will generate default names for features and label. </param>
        /// <returns>A list of instances parsed from the CSV file. </returns>
        public static List<Instance> ReadFromCsv(string filename, Range columns, string? featureTypes = null, bool hasHeader = false, string? headerNameList = null)
        {
            IEnumerable<string> dataRaw = File.ReadLines(filename);
            int actualColumnCount = columns.GetOffsetAndLength(dataRaw.First().Split(',').Count()).Length;
            List<Instance> instances = new List<Instance>();
            string[] headersParsed = new string[actualColumnCount];

            if (hasHeader)
            {
                string[] headerRowFields = dataRaw.First().Split(',')[columns];
                if (headerRowFields.Count() != headerRowFields.Distinct().Count())
                {
                    Debug.Fail("Duplicated values found in header row, checking provided headerNameList... ");
                    string[] headerNameListFields = headerNameList.Split(',');
                    if (headerNameListFields.Count() != actualColumnCount || headerNameListFields.Count() != headerNameListFields.Distinct().Count())
                    {
                        Debug.Fail("Incorrect string segments or duplicated values found in headerNameList, generating default headers... ");
                        // Generate default headers
                        for (int i = 0; i < headersParsed.Length - 1; i++)
                        {
                            headersParsed[i] = $"feature{i}";
                        }
                        headersParsed[^1] = "label";
                    }
                    else
                    {
                        // Use header name list as headers
                        headersParsed = headerNameListFields;
                    }
                }
                else
                {
                    // Use header row as headers
                    headersParsed = headerRowFields;
                }
            }
            else
            {
                // Generate default headers
                for (int i = 0; i < headersParsed.Length - 1; i++)
                {
                    headersParsed[i] = $"feature{i}";
                }
                headersParsed[^1] = "label";
            }

            foreach (string rowRaw in dataRaw)
            {
                if (hasHeader)
                {
                    hasHeader = false;
                    continue;
                }
                if (string.IsNullOrWhiteSpace(rowRaw))
                {
                    continue;
                }

                string[] fields = rowRaw.Split(',')[columns];
                // formulate new instance from fields
                instances.Add(new Instance(headersParsed, fields, featureTypes ?? string.Concat(Enumerable.Repeat('c', actualColumnCount - 1))));
            }
            return instances;
        }

        /// <summary>
        /// Parse a CSV file and return a table of string values. 
        /// </summary>
        /// <param name="filename">Path and filename of the CSV file to be parsed. </param>
        /// <param name="skipFirstRow">Whether the first row of the CSV file should be skipped. If true, the table will only include values starting from the second row. </param>
        /// <returns>A table of string values parsed from the CSV file. </returns>
        public static Table<string> ReadFromCsv(string filename, bool skipFirstRow = false) => ReadFromCsv(filename, .., skipFirstRow);

        /// <summary>
        /// Parse a range of columns of a CSV file and return a table of string values. 
        /// </summary>
        /// <param name="filename">Path and filename of the CSV file to be parsed. </param>
        /// <param name="columns">Range of columns in the CSV file to be parsed. </param>
        /// <param name="skipFirstRow">Whether the first row of the CSV file should be skipped. If true, the table will only include values starting from the second row. </param>
        /// <returns>A table of string values parsed from the CSV file. </returns>
        public static Table<string> ReadFromCsv(string filename, Range columns, bool skipFirstRow = false)
        {
            IEnumerable<string> dataRaw = File.ReadLines(filename);
            List<List<string>> dataParsed = new List<List<string>>();
            foreach (string rowRaw in dataRaw)
            {
                if (skipFirstRow)
                {
                    skipFirstRow = false;
                    continue;
                }
                if (string.IsNullOrWhiteSpace(rowRaw))
                {
                    continue;
                }
                string[] fields = rowRaw.Split(',')[columns];
                List<string> rowParsed = new List<string>();
                for (int i = 0; i < fields.Length; i++)
                {
                    rowParsed.Add(fields[i]);
                }
                dataParsed.Add(rowParsed);
            }
            return new Table<string>(dataParsed);
        }

        /// <summary>
        /// Write a table of string values into a CSV file.
        /// </summary>
        /// <param name="filename">Path and filename of the CSV file to be written into. </param>
        /// <param name="table">The table of string values to be written. </param>
        /// <param name="header">Headers of the columns of the table, separated by commas. If null, the method will not write a header row at the start of the file. </param>
        /// <param name="startIndex">Starting value of the index column. If null, the method will not write an index column. </param>
        public static void WriteToCsv(string filename, Table<string> table, string? header = null, int? startIndex = null)
        {
            if (header != null && header.Split(',').Count() != table.ColumnCount)
            {
                throw new ArgumentException($"Incorrect number of string segments in argument {header}");
            }

            List<string> rows = new List<string>();
            if (startIndex.HasValue)
            {
                if (header != null)
                {
                    rows.Add("Index," + header);
                }
                int currentIndex = startIndex.Value;
                foreach (List<string> rowData in table)
                {
                    StringBuilder sb = new StringBuilder(currentIndex++);
                    foreach (string field in rowData)
                    {
                        sb.Append("," + field);
                    }
                    rows.Add(sb.ToString());
                }
            }
            else
            {
                if (header != null)
                {
                    rows.Add(header);
                }
                foreach (List<string> rowData in table)
                {
                    StringBuilder sb = new StringBuilder("");
                    foreach (string field in rowData)
                    {
                        sb.Append("," + field);
                    }
                    rows.Add(sb.ToString()[1..]);
                }
            }
            File.WriteAllLines(filename, rows);
        }

        /// <summary>
        /// Write a list of instances into a CSV file.
        /// </summary>
        /// <param name="filename">Path and filename of the CSV file to be written into. </param>
        /// <param name="instances">The list of instances to be written. </param>
        /// <param name="writeHeader">Whether the name of features and label will be written as the header row at the start of the file. </param>
        /// <param name="startIndex">Starting value of the index column. If null, the method will not write an index column. </param>
        public static void WriteToCsv(string filename, List<Instance> instances, bool writeHeader = false, int? startIndex = null)
        {
            List<string> rows = new List<string>();

            if (writeHeader)
            {
                StringBuilder header = new StringBuilder(startIndex.HasValue ? "Index," : "");
                foreach (Feature feature in instances.First().Features)
                {
                    header.Append(feature.Name + ",");
                }
                header.Append(instances.First().LabelName);
                rows.Add(header.ToString());
            }

            if (startIndex.HasValue)
            {
                int currentIndex = startIndex.Value;
                foreach (Instance instance in instances)
                {
                    StringBuilder sb = new StringBuilder($"{currentIndex++},");
                    foreach (Feature feature in instance.Features)
                    {
                        sb.Append(feature.Value + ",");
                    }
                    sb.Append(instance.LabelValue);
                    rows.Add(sb.ToString());
                }
            }
            else
            {
                foreach (Instance instance in instances)
                {
                    StringBuilder sb = new StringBuilder();
                    foreach (Feature feature in instance.Features)
                    {
                        sb.Append(feature.Value + ",");
                    }
                    sb.Append(instance.LabelValue);
                    rows.Add(sb.ToString());
                }
            }

            File.WriteAllLines(filename, rows);
        }
    }
}
