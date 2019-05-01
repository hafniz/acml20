using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLCore
{
    public enum ValueType
    {
        Continuous,
        Discrete
    }

    public class Feature
    {
        public ValueType ValueType { get; set; }
        public dynamic Value { get; set; }

        private Feature() => throw new InvalidOperationException();
        public Feature(ValueType valueType, dynamic value)
        {
            ValueType = valueType;
            Value = value;
        }
    }

    public class Instance : ICloneable
    {
        public Dictionary<string, Feature> Features { get; private set; }
        public string? LabelValue { get; private set; }
        public string? LabelName { get; private set; }

        private Instance() => throw new InvalidOperationException();
        public Instance(Dictionary<string, Feature> features, string? labelValue = null, string? labelName = null)
        {
            Features = features;
            LabelValue = labelValue;
            LabelName = labelName;
        }
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder($"Instance ({LabelName ?? "label"}: {LabelValue ?? "unlabeled"})\n");
            foreach (KeyValuePair<string, Feature> kvp in Features)
            {
                sb.Append($" {kvp.Key}: {kvp.Value.Value}\n");
            }
            return sb.ToString();
        }
        public string Serialize()
        {
            StringBuilder sb = new StringBuilder("");
            foreach (dynamic featureValue in Features.Select(kvp => kvp.Value.Value))
            {
                sb.Append(featureValue.ToString() + ",");
            }
            sb.Append(LabelValue);
            return sb.ToString();
        }

        public object Clone()
        {
            Instance newInstance = new Instance(new Dictionary<string, Feature>(), LabelValue, LabelName);
            foreach (KeyValuePair<string, Feature> kvp in Features)
            {
                newInstance.Features.Add(kvp.Key, new Feature(kvp.Value.ValueType, kvp.Value.Value));
            }
            return newInstance;
        }
    }
}
