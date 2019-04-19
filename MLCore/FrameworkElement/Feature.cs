namespace MLCore.FrameworkElement
{
    public struct Feature
    {
        public ValueType ValueType { get; private set; }
        public dynamic Value { get; private set; }
        public string Name { get; private set; }

        public Feature(ValueType valueType, dynamic value, string name = "unnamed feature")
        {
            ValueType = valueType;
            Value = value;
            Name = name;
        }
        public override string ToString() => $"{Name}: {Value}";
    }
}
