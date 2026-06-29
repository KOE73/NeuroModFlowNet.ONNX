namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Defines how coordinate payload mappers may handle transforms that can change the geometric shape class.
/// </summary>
public enum CoordinateMappingShapePolicy
{
    PreserveShape,
    BoundingBox,
    BoundingOBB,
    Quad,
    RejectIfShapeChanges
}
