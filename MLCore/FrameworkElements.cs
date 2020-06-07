using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

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
    public class Instance : ICloneable, IEquatable<Instance>
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

        public override string ToString() => $"Instance ({LabelName ?? "label"}: {LabelValue ?? "unlabeled"}) {string.Join(' ', Features.Select(f => $"{f.Name}: {f.Value}{(f.ValueDiscretized is null ? "" : $" ({f.ValueDiscretized})")}"))}";

        public string Serialize() => $"{string.Join(',', Features.Select(f => f.Value.ToString()))},{LabelValue}";

        public object Clone()
        {
            Instance newInstance = new Instance(new List<Feature>(), LabelValue, LabelName);
            Features.ForEach(f => newInstance.Features.Add(new Feature(f.Name, f.ValueType, f.Value)));
            return newInstance;
        }

        public bool Equals(Instance? other)
        {
            if (other is null)
            {
                return this is null;
            }
            if (other.LabelValue != LabelValue)
            {
                return false;
            }
            if (other.Features.Count != Features.Count)
            {
                return false;
            }
            List<string> otherInstanceFeatureNames = other.Features.Select(f => f.Name).ToList();
            foreach (Feature f in Features)
            {
                if (!otherInstanceFeatureNames.Contains(f.Name))
                {
                    return false;
                }
                if (other[f.Name].Value != f.Value)
                {
                    return false;
                }
            }
            return true;
        }
    }
}
