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
        /// Read and parse the content of a CSV file into a list of instances. 
        /// </summary>
        /// <param name="filename">Relative or absolute path of the CSV file to be read, including filename and file extension. </param>
        /// <param name="featureTypes">A string specifying the ValueType of each feature (NOT including the label and index, if present), each feature represented by a char: 'c' for ValueType.Continuous, 'd' for ValueType.Discrete. Leaving as null is equivalent to marking all feature as continuous. </param>
        /// <param name="headerNameList">A string specifying name of each feature and label (NOT including the index, if present), separated by commas. </param>
        /// <param name="hasIndex">Specifies whether the first column of the file is the index of instance, which has no correlation with the label. If true, the first column of the file will be ignored. </param>
        /// <param name="trimRight">Number of rightmost columns to be ignored when parsing the file. </param>
        /// <returns>A list of instances parsed from the CSV file. </returns>
        public static List<Instance> ReadFromCsv(string filename, string? featureTypes = null, string? headerNameList = null, bool hasIndex = false, int trimRight = 0)
        {
            static bool AreDistinct(object[] values)
            {
                List<object> list = new List<object>();
                foreach (object value in values)
                {
                    if (list.Contains(value))
                    {
                        return false;
                    }
                    list.Add(value);
                }
                return true;
            }

            string[]? headerNames = headerNameList?.Split(',');
            if (headerNames != null && !AreDistinct(headerNames))
            {
                throw new ArgumentException("Duplicate header names found in input string. ");
            }

            List<Instance> instances = new List<Instance>();
            IEnumerable<string> rows = File.ReadLines(filename);

            int columnCount = rows.First().Split(',').Count() - trimRight;
            int effectiveColumnCount = hasIndex ? columnCount - 1 : columnCount;
            int featureCount = effectiveColumnCount - 1;

            if (headerNames != null && headerNames.Length != effectiveColumnCount)
            {
                throw new ArgumentException($"Incorrect number of string segments in argument {nameof(headerNameList)}. ");
            }
            if (featureTypes != null && featureTypes?.Count() != featureCount)
            {
                throw new ArgumentException($"Incorrect number of characters in argument {nameof(featureTypes)}. ");
            }

            bool hasHeader = !(headerNames is null);
            foreach (string row in rows)
            {
                if (hasHeader)
                {
                    hasHeader = false;
                    continue;
                }

                string[] fields = row.Split(',')[..^trimRight];
                Dictionary<string, Feature> features = new Dictionary<string, Feature>();

                for (int featureNumber = 0; featureNumber < featureCount; featureNumber++)
                {
                    int columnNumber = hasIndex ? featureNumber + 1 : featureNumber;
                    ValueType valueType = featureTypes is null || char.ToUpper(featureTypes[featureNumber]) == 'C' ? ValueType.Continuous : ValueType.Discrete;
                    string featureName = headerNames is null || string.IsNullOrWhiteSpace(headerNames[featureNumber]) ? $"feature{featureNumber}" : headerNames[featureNumber];
                    if (valueType == ValueType.Continuous)
                    {
                        features.Add(featureName, new Feature(valueType, double.Parse(fields[columnNumber])));
                    }
                    else
                    {
                        features.Add(featureName, new Feature(valueType, fields[columnNumber]));
                    }
                }

                instances.Add(new Instance(features, fields[^1], string.IsNullOrWhiteSpace(headerNames?[^1]) ? "label" : headerNames[^1]));
            }
            return instances;
        }

        /// <summary>
        /// Read and parse the content of a CSV file into a list of list of string values. 
        /// </summary>
        /// <param name="filename">Relative or absolute path of the CSV file to be read, including filename and file extension. </param>
        /// <param name="hasHeader">Specifies whether the first row of the file is the header rather than actual value of a row of fields. If true, the first row of the file will be ignored. </param>
        /// <param name="hasIndex">Specifies whether the first column of the file is the index of rows. If true, the first column of the file will be ignored. </param>
        /// <returns>A list of list of string values parsed from the CSV file. </returns>
        public static Table<string> ReadFromCsv(string filename, bool hasHeader = false, bool hasIndex = false)
        {
            IEnumerable<string> rows = File.ReadLines(filename);
            List<List<string>> data = new List<List<string>>();
            foreach (string row in rows)
            {
                if (hasHeader)
                {
                    hasHeader = false;
                    continue;
                }
                string[] fields = row.Split(',');
                List<string> rowData = new List<string>();
                for (int i = hasIndex ? 1 : 0; i < fields.Length; i++)
                {
                    rowData.Add(fields[i]);
                }
                data.Add(rowData);
            }
            return new Table<string>(data);
        }

        /// <summary>
        /// Write data into a CSV file.
        /// </summary>
        /// <param name="filename">Relative or absolute path of the CSV file to write, including filename and file extension. </param>
        /// <param name="table">Data to be written into the file. Each List&lt;object&gt; represents a row while each object in the List represents a field. </param>
        /// <param name="header">Header of the data excluding index, separated by commas. Leaving as null will result in the header row not being written. </param>
        /// <param name="writeIndex">Specifies whether a zero-based index is written at the beginning of each row. The header of this column, if specified to be written, is 'Index'. </param>
        public static void WriteToCsv(string filename, Table<string> table, string? header = null, bool writeIndex = false)
        {
            if (header != null && header.Split(',').Count() != table.ColumnCount)
            {
                throw new ArgumentException($"Incorrect number of string segments in argument {header}");
            }

            List<string> rows = new List<string>();
            if (writeIndex)
            {
                if (header != null)
                {
                    rows.Add("Index," + header);
                }
                int index = 0;
                foreach (List<string> rowData in table)
                {
                    StringBuilder sb = new StringBuilder(index++);
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
                    rows.Add(sb.ToString().Substring(1));
                }
            }
            File.WriteAllLines(filename, rows);
        }

        /// <summary>
        /// Write instances into a CSV file.
        /// </summary>
        /// <param name="filename">Relative or absolute path of the CSV file to write, including filename and file extension. </param>
        /// <param name="instances">Instances to be written into the file. </param>
        /// <param name="writeHeader">Specifies whether the name of each feature and label should be recorded as the header row at the top of the file. </param>
        /// <param name="writeIndex">Specifies whether a index is written at the beginning of each row. The header of this column, if specified to be written, is 'Index'. </param>
        public static void WriteToCsv(string filename, List<Instance> instances, bool writeHeader = false, bool writeIndex = false)
        {
            List<string> rows = new List<string>();

            if (writeHeader)
            {
                StringBuilder header = new StringBuilder(writeIndex ? "Index," : "");
                foreach (KeyValuePair<string, Feature> kvp in instances[0].Features)
                {
                    header.Append(kvp.Key + ",");
                }
                header.Append(instances[0].LabelName);
                rows.Add(header.ToString());
            }

            int index = 0;
            foreach (Instance instance in instances)
            {
                StringBuilder sb = new StringBuilder(writeIndex ? $"{index++}," : "");
                foreach (KeyValuePair<string, Feature> kvp in instance.Features)
                {
                    sb.Append(kvp.Value.Value + ",");
                }
                sb.Append(instance.LabelValue);
                rows.Add(sb.ToString());
            }

            File.WriteAllLines(filename, rows);
        }
    }
}
