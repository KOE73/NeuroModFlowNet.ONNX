using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

internal static class PerspectiveHomography
{
    public static PerspectiveCoordinateBackTransform CreateTargetToSourceTransform(
        ReadOnlySpan<Point2f> sourcePoints,
        int outputWidth,
        int outputHeight)
    {
        Span<float> matrix = stackalloc float[9];
        WriteTargetToSourceMatrix(sourcePoints, outputWidth, outputHeight, matrix);
        return new PerspectiveCoordinateBackTransform(
            matrix[0], matrix[1], matrix[2],
            matrix[3], matrix[4], matrix[5],
            matrix[6], matrix[7], matrix[8]);
    }

    public static void WriteTargetToSourceMatrix(
        ReadOnlySpan<Point2f> sourcePoints,
        int outputWidth,
        int outputHeight,
        Span<float> matrix)
    {
        if(sourcePoints.Length < 4)
            throw new ArgumentException("At least four source points are required.", nameof(sourcePoints));

        if(matrix.Length < 9)
            throw new ArgumentException("Matrix destination must contain 9 values.", nameof(matrix));

        Span<Point2f> targetPoints = stackalloc Point2f[4];
        targetPoints[0] = new Point2f(0, 0);
        targetPoints[1] = new Point2f(outputWidth - 1, 0);
        targetPoints[2] = new Point2f(outputWidth - 1, outputHeight - 1);
        targetPoints[3] = new Point2f(0, outputHeight - 1);

        SolveHomography(targetPoints, sourcePoints, matrix);
    }

    static void SolveHomography(
        ReadOnlySpan<Point2f> sourcePoints,
        ReadOnlySpan<Point2f> destinationPoints,
        Span<float> matrix)
    {
        Span<double> augmented = stackalloc double[8 * 9];

        for(int pointIndex = 0; pointIndex < 4; pointIndex++)
        {
            double x = sourcePoints[pointIndex].X;
            double y = sourcePoints[pointIndex].Y;
            double u = destinationPoints[pointIndex].X;
            double v = destinationPoints[pointIndex].Y;
            int row = pointIndex * 2;

            WriteRow(augmented, row, x, y, 1, 0, 0, 0, -u * x, -u * y, u);
            WriteRow(augmented, row + 1, 0, 0, 0, x, y, 1, -v * x, -v * y, v);
        }

        SolveLinearSystem(augmented);

        matrix[0] = (float)augmented[8];
        matrix[1] = (float)augmented[17];
        matrix[2] = (float)augmented[26];
        matrix[3] = (float)augmented[35];
        matrix[4] = (float)augmented[44];
        matrix[5] = (float)augmented[53];
        matrix[6] = (float)augmented[62];
        matrix[7] = (float)augmented[71];
        matrix[8] = 1f;
    }

    static void WriteRow(
        Span<double> augmented,
        int row,
        double a0,
        double a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double value)
    {
        int offset = row * 9;
        augmented[offset] = a0;
        augmented[offset + 1] = a1;
        augmented[offset + 2] = a2;
        augmented[offset + 3] = a3;
        augmented[offset + 4] = a4;
        augmented[offset + 5] = a5;
        augmented[offset + 6] = a6;
        augmented[offset + 7] = a7;
        augmented[offset + 8] = value;
    }

    static void SolveLinearSystem(Span<double> augmented)
    {
        const int order = 8;
        const int stride = 9;

        for(int pivotIndex = 0; pivotIndex < order; pivotIndex++)
        {
            int bestRow = pivotIndex;
            double bestValue = Math.Abs(augmented[pivotIndex * stride + pivotIndex]);
            for(int row = pivotIndex + 1; row < order; row++)
            {
                double value = Math.Abs(augmented[row * stride + pivotIndex]);
                if(value > bestValue)
                {
                    bestValue = value;
                    bestRow = row;
                }
            }

            if(bestValue <= double.Epsilon)
                throw new InvalidOperationException("Perspective source points produce a singular homography.");

            if(bestRow != pivotIndex)
                SwapRows(augmented, pivotIndex, bestRow);

            double pivot = augmented[pivotIndex * stride + pivotIndex];
            for(int column = pivotIndex; column < stride; column++)
                augmented[pivotIndex * stride + column] /= pivot;

            for(int row = 0; row < order; row++)
            {
                if(row == pivotIndex)
                    continue;

                double factor = augmented[row * stride + pivotIndex];
                for(int column = pivotIndex; column < stride; column++)
                    augmented[row * stride + column] -= factor * augmented[pivotIndex * stride + column];
            }
        }
    }

    static void SwapRows(Span<double> augmented, int firstRow, int secondRow)
    {
        const int stride = 9;
        for(int column = 0; column < stride; column++)
        {
            int firstIndex = firstRow * stride + column;
            int secondIndex = secondRow * stride + column;
            (augmented[firstIndex], augmented[secondIndex]) = (augmented[secondIndex], augmented[firstIndex]);
        }
    }
}
