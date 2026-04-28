using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

namespace NeuroModFlowNet.ONNX.Tools.Modify;

[AttributeUsage(AttributeTargets.Class)]
public sealed class OnnxHeadAttribute : Attribute
{
    public OnnxHeadAttribute(string name, string description)
    {
        Name = name;
        Description = description;
    }
    public string Name { get; }
    public string Description { get; }
    public string? Example { get; set; }


}


public static class OnnxHeadRegistry
{
    // Map: Name -> Type
    private static readonly Dictionary<string, (Type Type, string Desc, string? Example)> _registry = new(StringComparer.OrdinalIgnoreCase);

    static OnnxHeadRegistry()
    {
        // Scan current assembly for classes inheriting from OnnxModelModifier with [OnnxHead] attribute
        var types = Assembly.GetExecutingAssembly().GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsSubclassOf(typeof(OnnxModelModifier)));

        foreach(var type in types)
        {
            var attr = type.GetCustomAttribute<OnnxHeadAttribute>();
            if(attr != null)
            {
                _registry[attr.Name] = (type, attr.Description, attr.Example);
            }
        }
    }

    public static IEnumerable<string> GetAvailableNames() => _registry.Keys;

    public static string GetDetailedHelp()
    {
        var lines = _registry.Select(kv => $"  [yellow]{kv.Key,-20}[/] [grey]{kv.Value.Desc}[/]");
        return $"Injects a specialized preprocessing head.\n\n[white]Available types:[/]\n{string.Join("\n", lines)}";
    }

    public static IEnumerable<string[]> GetExamples() =>
        _registry.Values
            .Where(v => v.Example is not null)
            .Select(v => v.Example!.Split(' '));

    public static OnnxModelModifier Create(string name)
    {
        if(!_registry.TryGetValue(name, out var item))
            throw new ArgumentException($"Head type '{name}' is not registered.");

        return (OnnxModelModifier)Activator.CreateInstance(item.Type)!;
    }
}