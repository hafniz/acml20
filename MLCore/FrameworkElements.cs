using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MLCore
{
    public enum ValueType
    {
        Continuous,
        Discrete
    }

    [DebuggerStepThrough]
    public class Feature
    {
        public string Name { get; }
        public ValueType ValueType { get; }
        public dynamic Value { get; }
        public string? ValueDiscretized { get; set; }
        private Feature() => throw new InvalidOperationException();
        public Feature(string name, ValueType valueType, dynamic value)
        {
            Name = name;
            ValueType = valueType;
            Value = value;
        }
    }

    [DebuggerStepThrough]
    public class Instance : ICloneable
    {
        public List<Feature> Features { get; }
        public string? LabelValue { get; }
        public string? LabelName { get; }
        public Feature this[string featureName] => Features.Where(f => f.Name == featureName).Single();

        private Instance() => throw new InvalidOperationException();
        public Instance(List<Feature> features, string? labelValue = null, string? labelName = null)
        {
            Features = features;
            LabelValue = labelValue;
            LabelName = labelName;
        }

        public Instance(string[] headers, string[] values, string featureTypes)
        {
            Features = new List<Feature>();
            for (int i = 0; i < headers.Length - 1; i++)
            {
                if (char.ToUpper(featureTypes[i]) == 'C')
                {
                    Features.Add(new Feature(headers[i], ValueType.Continuous, double.Parse(values[i])));
                }
                else
                {
                    Features.Add(new Feature(headers[i], ValueType.Discrete, values[i]));
                }
            }
            LabelValue = values[^1];
            LabelName = headers[^1];
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder($"Instance ({LabelName ?? "label"}: {LabelValue ?? "unlabeled"})\n");
            foreach (Feature feature in Features)
            {
                sb.Append($" {feature.Name}: {feature.Value}{(feature.ValueDiscretized is null ? "" : " (" + feature.ValueDiscretized + ")")}\n");
            }
            return sb.ToString();
        }

        public string Serialize()
        {
            StringBuilder sb = new StringBuilder("");
            foreach (dynamic featureValue in Features.Select(f => f.Value))
            {
                sb.Append(featureValue.ToString() + ",");
            }
            sb.Append(LabelValue);
            return sb.ToString();
        }

        public object Clone()
        {
            Instance newInstance = new Instance(new List<Feature>(), LabelValue, LabelName);
            foreach (Feature feature in Features)
            {
                newInstance.Features.Add(new Feature(feature.Name, feature.ValueType, feature.Value));
            }
            return newInstance;
        }
    }
}
