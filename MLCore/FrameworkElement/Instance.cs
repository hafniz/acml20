using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.VisualBasic.FileIO;

namespace MLCore.FrameworkElement
{
    public struct Instance
    {
        public List<Feature> Features { get; private set; }
        public dynamic? Label { get; private set; }
        public Instance(List<Feature> features, dynamic? label = null)
        {
            Features = features;
            Label = label;
        }
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder($"Instance ({Label ?? "unlabeled"}), ");
            foreach (Feature feature in Features)
            {
                sb.Append(" " + feature);
            }
            return sb.ToString();
        }

        /// <summary>
        /// Read and parse the content of a CSV file into a list of instances. 
        /// </summary>
        /// <param name="filename">Relative or absolute path of the CSV file, including filename and file extension. </param>
        /// <param name="featureProtypes">Specifications for how the value of features should be parsed. 
        /// This indicates whether the value is continuous or discrete, type of value (by entering a sample value) and the name, if specified, for each feature. </param>
        /// <param name="labelPrototype">Specifications for how the value of label should be parsed. 
        /// This indicates the type of value (by entering a sample value) of the label. </param>
        /// <param name="hasHeader">Specifies whether the first row of the content is the header rather than actual value of an instance. 
        /// If true, the first row of the content will be ignored. </param>
        /// <param name="hasIndex">Specifies whether the first column of the content is the index of instance, which has no correlation with the label. 
        /// If true, the first column of the content will be ignored. </param>
        /// <returns>A list of instances parsed from the CSV file. </returns>
        public static List<Instance> ReadFromCsv(string filename, List<Feature> featureProtypes, dynamic labelPrototype, bool hasHeader = false, bool hasIndex = false)
        {
            List<Instance> instances = new List<Instance>();
            using (TextFieldParser parser = new TextFieldParser(filename))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");
                while (!parser.EndOfData)
                {
                    if (hasHeader)
                    {
                        hasHeader = false;
                        continue;
                    }
                    List<Feature> features = new List<Feature>();
                    string[] row = parser.ReadFields();
                    for (int i = hasIndex ? 1 : 0; i < row.Length - 1; i++)
                    {
                        Feature prototype = featureProtypes[hasIndex ? i - 1 : i];
                        features.Add(new Feature(prototype.ValueType, Convert.ChangeType(row[i], prototype.Value.GetType()), prototype.Name));
                    }
                    instances.Add(new Instance(features, Convert.ChangeType(row[^1], labelPrototype.GetType())));
                }
            }
            return instances;
        }
    }
}
