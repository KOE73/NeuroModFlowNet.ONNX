using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Realtime-oriented geometry postprocessor for OCR quadrilateral regions.
/// RU: Realtime-ориентированная геометрическая постобработка четырехточечных OCR-регионов.
/// </summary>
/// <remarks>
/// EN: The processor deliberately owns the cached geometry for one call. Filters, overlap cleanup and optional
/// line merge all use the same candidate data instead of recomputing width, height, axis and area in separate
/// pipeline stages. This keeps the hot path explicit and makes feature switches cheap.
/// RU: Процессор намеренно владеет кэшированной геометрией одного вызова. Фильтры, удаление пересечений и
/// опциональная склейка строк используют одни и те же данные кандидата, а не пересчитывают width, height, axis
/// и area в отдельных pipeline stages. Это сохраняет hot path явным и делает переключатели фич дешевыми.
/// </remarks>
public sealed class OcrRegionPostprocessor
{
    public static OcrRegionPostprocessor Shared { get; } = new();

    #region Public API

    public int Process(
        ReadOnlySpan<OcrQuadRegion> sourceRegions,
        in OcrRegionPostprocessorOptions options,
        List<OcrQuadRegion> destinationRegions)
    {
        ArgumentNullException.ThrowIfNull(destinationRegions);
        ValidateOptions(options);

        if(sourceRegions.Length == 0) return 0;

        var candidates = new List<RegionCandidate>(sourceRegions.Length);
        for(int index = 0; index < sourceRegions.Length; index++)
        {
            if(!TryCreateCandidate(sourceRegions[index], out RegionCandidate candidate)) continue;
            if(!PassesDomainFilters(candidate, options)) continue;

            candidates.Add(candidate);
        }

        if(options.EnableOverlapSuppression && candidates.Count > 1)
            SuppressOverlaps(candidates, options);

        if(options.EnableLineMerge && candidates.Count > 1)
            MergeLineCandidates(candidates, options);

        int resultCount = Math.Min(candidates.Count, options.MaxRegions);
        for(int index = 0; index < resultCount; index++)
            destinationRegions.Add(candidates[index].Region);

        return resultCount;
    }

    #endregion

    #region Candidate Filtering

    private static void ValidateOptions(in OcrRegionPostprocessorOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(options.MinRegionHeight);
        ArgumentOutOfRangeException.ThrowIfNegative(options.MinRegionWidth);
        ArgumentOutOfRangeException.ThrowIfNegative(options.MinRegionAspectRatio);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.OverlapSuppressionRatio);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MergeHeightRatio);
        ArgumentOutOfRangeException.ThrowIfNegative(options.MaxRegions);
    }

    private static bool PassesDomainFilters(in RegionCandidate candidate, in OcrRegionPostprocessorOptions options)
    {
        if(candidate.Height < options.MinRegionHeight || candidate.Height > options.MaxRegionHeight) return false;
        if(candidate.Width < options.MinRegionWidth || candidate.Width > options.MaxRegionWidth) return false;
        if(candidate.AspectRatio < options.MinRegionAspectRatio || candidate.AspectRatio > options.MaxRegionAspectRatio) return false;
        return true;
    }

    private static bool TryCreateCandidate(OcrQuadRegion region, out RegionCandidate candidate)
    {
        Span<Point2f> points = stackalloc Point2f[4];
        region.CopyTo(points);

        if(!TryNormalizePoints(points))
        {
            candidate = default;
            return false;
        }

        float topWidth = Distance(points[0], points[1]);
        float bottomWidth = Distance(points[3], points[2]);
        float leftHeight = Distance(points[0], points[3]);
        float rightHeight = Distance(points[1], points[2]);
        float width = MathF.Max(topWidth, bottomWidth);
        float height = MathF.Max(leftHeight, rightHeight);
        float area = MathF.Abs(GetPolygonArea(points));

        if(width <= float.Epsilon || height <= float.Epsilon || area <= float.Epsilon)
        {
            candidate = default;
            return false;
        }

        Point2f center = new(
            (points[0].X + points[1].X + points[2].X + points[3].X) * 0.25f,
            (points[0].Y + points[1].Y + points[2].Y + points[3].Y) * 0.25f);

        Point2f axis = Normalize(new Point2f(points[1].X - points[0].X, points[1].Y - points[0].Y));
        if(axis.X < 0 || axis.X == 0 && axis.Y < 0)
            axis = new Point2f(-axis.X, -axis.Y);

        candidate = new RegionCandidate(
            OcrQuadRegion.FromPoints(points),
            points[0],
            points[1],
            points[2],
            points[3],
            center,
            axis,
            width,
            height,
            area);
        return true;
    }

    private static bool TryNormalizePoints(Span<Point2f> points)
    {
        Point2f topLeft = points[0];
        Point2f bottomRight = points[0];
        Point2f topRight = points[0];
        Point2f bottomLeft = points[0];

        for(int index = 1; index < points.Length; index++)
        {
            Point2f point = points[index];
            if(point.X + point.Y < topLeft.X + topLeft.Y) topLeft = point;
            if(point.X + point.Y > bottomRight.X + bottomRight.Y) bottomRight = point;
            if(point.X - point.Y > topRight.X - topRight.Y) topRight = point;
            if(point.X - point.Y < bottomLeft.X - bottomLeft.Y) bottomLeft = point;
        }

        points[0] = topLeft;
        points[1] = topRight;
        points[2] = bottomRight;
        points[3] = bottomLeft;

        return GetPolygonArea(points) > float.Epsilon;
    }

    #endregion

    #region Overlap Suppression

    private static void SuppressOverlaps(List<RegionCandidate> candidates, in OcrRegionPostprocessorOptions options)
    {
        for(int leftIndex = 0; leftIndex < candidates.Count; leftIndex++)
        {
            RegionCandidate left = candidates[leftIndex];

            for(int rightIndex = candidates.Count - 1; rightIndex > leftIndex; rightIndex--)
            {
                RegionCandidate right = candidates[rightIndex];
                if(!AabbIntersects(left, right)) continue;
                if(!PolygonsIntersect(left, right)) continue;

                float smallerArea = MathF.Min(left.Area, right.Area);
                float overlapProxy = EstimateOverlapProxy(left, right);
                if(smallerArea <= float.Epsilon || overlapProxy / smallerArea < options.OverlapSuppressionRatio)
                    continue;

                // EN: At this stage OCR regions no longer carry detector score. Keeping the larger region is the
                // least surprising fallback: it usually preserves the full text line when duplicate boxes overlap.
                // RU: На этом этапе OCR-регионы уже не несут score детектора. Оставляем больший регион как самый
                // предсказуемый вариант: при дублирующихся пересечениях он чаще сохраняет всю строку текста.
                if(left.Area >= right.Area)
                {
                    candidates.RemoveAt(rightIndex);
                }
                else
                {
                    candidates.RemoveAt(leftIndex);
                    leftIndex--;
                    break;
                }
            }
        }
    }

    private static float EstimateOverlapProxy(in RegionCandidate left, in RegionCandidate right)
    {
        float x1 = MathF.Max(left.MinX, right.MinX);
        float y1 = MathF.Max(left.MinY, right.MinY);
        float x2 = MathF.Min(left.MaxX, right.MaxX);
        float y2 = MathF.Min(left.MaxY, right.MaxY);
        return MathF.Max(0, x2 - x1) * MathF.Max(0, y2 - y1);
    }

    private static bool AabbIntersects(in RegionCandidate left, in RegionCandidate right) =>
        left.MinX <= right.MaxX &&
        left.MaxX >= right.MinX &&
        left.MinY <= right.MaxY &&
        left.MaxY >= right.MinY;

    private static bool PolygonsIntersect(in RegionCandidate left, in RegionCandidate right)
    {
        Span<Point2f> axes = stackalloc Point2f[4];
        axes[0] = GetNormal(left.Point1, left.Point0);
        axes[1] = GetNormal(left.Point2, left.Point1);
        axes[2] = GetNormal(right.Point1, right.Point0);
        axes[3] = GetNormal(right.Point2, right.Point1);

        for(int index = 0; index < axes.Length; index++)
        {
            Point2f axis = Normalize(axes[index]);
            ProjectionInterval leftInterval = Project(left, axis);
            ProjectionInterval rightInterval = Project(right, axis);
            if(leftInterval.Maximum < rightInterval.Minimum || rightInterval.Maximum < leftInterval.Minimum)
                return false;
        }

        return true;
    }

    #endregion

    #region Line Merge

    private static void MergeLineCandidates(List<RegionCandidate> candidates, in OcrRegionPostprocessorOptions options)
    {
        bool mergedAnyCandidate;
        do
        {
            mergedAnyCandidate = false;

            for(int leftIndex = 0; leftIndex < candidates.Count && !mergedAnyCandidate; leftIndex++)
            {
                for(int rightIndex = leftIndex + 1; rightIndex < candidates.Count; rightIndex++)
                {
                    if(!CanMerge(candidates[leftIndex], candidates[rightIndex], options, out RegionCandidate merged))
                        continue;

                    candidates[leftIndex] = merged;
                    candidates.RemoveAt(rightIndex);
                    mergedAnyCandidate = true;
                    break;
                }
            }
        }
        while(mergedAnyCandidate);
    }

    private static bool CanMerge(
        in RegionCandidate left,
        in RegionCandidate right,
        in OcrRegionPostprocessorOptions options,
        out RegionCandidate merged)
    {
        merged = default;

        Point2f sharedAxis = left.Area >= right.Area ? left.Axis : right.Axis;
        Point2f sharedNormal = new(-sharedAxis.Y, sharedAxis.X);
        Point2f centerDelta = new(right.Center.X - left.Center.X, right.Center.Y - left.Center.Y);
        float meanHeight = (left.Height + right.Height) * 0.5f;
        if(meanHeight <= float.Epsilon) return false;

        if(GetAngleDeltaDegrees(left.Axis, right.Axis) > options.MergeAngleDeltaDegrees) return false;

        float normalizedNormalOffset = MathF.Abs(Dot(centerDelta, sharedNormal)) / meanHeight;
        if(normalizedNormalOffset > options.MergeNormalOffsetInHeights) return false;

        float heightRatio = MathF.Max(left.Height, right.Height) / MathF.Min(left.Height, right.Height);
        if(heightRatio > options.MergeHeightRatio) return false;

        float gapAlongAxis = GetGapAlongAxis(left, right, sharedAxis);
        if(gapAlongAxis / meanHeight > options.MergeGapInHeights) return false;

        if(!TryBuildMergedCandidate(left, right, out merged)) return false;
        if(merged.Width > options.MaxMergedRegionWidth) return false;
        if(merged.AspectRatio > options.MaxMergedRegionAspectRatio) return false;

        float coverageRatio = (left.Area + right.Area) / merged.Area;
        if(coverageRatio < options.MinimumMergedCoverageRatio) return false;

        return true;
    }

    private static bool TryBuildMergedCandidate(
        in RegionCandidate left,
        in RegionCandidate right,
        out RegionCandidate merged)
    {
        Point2f[] allPoints =
        [
            left.Point0,
            left.Point1,
            left.Point2,
            left.Point3,
            right.Point0,
            right.Point1,
            right.Point2,
            right.Point3,
        ];

        RotatedRect rect = Cv2.MinAreaRect(allPoints);
        Point2f[] boxPoints = rect.Points();
        return TryCreateCandidate(OcrQuadRegion.FromPoints(boxPoints), out merged);
    }

    private static float GetGapAlongAxis(in RegionCandidate left, in RegionCandidate right, Point2f axis)
    {
        ProjectionInterval leftInterval = Project(left, axis);
        ProjectionInterval rightInterval = Project(right, axis);

        if(leftInterval.Maximum < rightInterval.Minimum)
            return rightInterval.Minimum - leftInterval.Maximum;

        if(rightInterval.Maximum < leftInterval.Minimum)
            return leftInterval.Minimum - rightInterval.Maximum;

        return 0f;
    }

    private static float GetAngleDeltaDegrees(Point2f leftAxis, Point2f rightAxis)
    {
        float dot = Math.Clamp(MathF.Abs(Dot(leftAxis, rightAxis)), 0f, 1f);
        return MathF.Acos(dot) * 180f / MathF.PI;
    }

    #endregion

    #region Geometry Helpers

    private static ProjectionInterval Project(in RegionCandidate candidate, Point2f axis)
    {
        float minimum = Dot(candidate.Point0, axis);
        float maximum = minimum;

        UpdateProjection(candidate.Point1, axis, ref minimum, ref maximum);
        UpdateProjection(candidate.Point2, axis, ref minimum, ref maximum);
        UpdateProjection(candidate.Point3, axis, ref minimum, ref maximum);

        return new ProjectionInterval(minimum, maximum);
    }

    private static void UpdateProjection(Point2f point, Point2f axis, ref float minimum, ref float maximum)
    {
        float projection = Dot(point, axis);
        if(projection < minimum) minimum = projection;
        if(projection > maximum) maximum = projection;
    }

    private static Point2f GetNormal(Point2f first, Point2f second) =>
        new(-(first.Y - second.Y), first.X - second.X);

    private static Point2f Normalize(Point2f vector)
    {
        float length = MathF.Sqrt(vector.X * vector.X + vector.Y * vector.Y);
        return length <= float.Epsilon
            ? new Point2f(1f, 0f)
            : new Point2f(vector.X / length, vector.Y / length);
    }

    private static float Distance(Point2f first, Point2f second)
    {
        float x = first.X - second.X;
        float y = first.Y - second.Y;
        return MathF.Sqrt(x * x + y * y);
    }

    private static float Dot(Point2f first, Point2f second) => first.X * second.X + first.Y * second.Y;

    private static float GetPolygonArea(ReadOnlySpan<Point2f> points)
    {
        float area = 0f;
        for(int index = 0; index < points.Length; index++)
        {
            Point2f current = points[index];
            Point2f next = points[(index + 1) % points.Length];
            area += current.X * next.Y - next.X * current.Y;
        }

        return area * 0.5f;
    }

    #endregion

    #region Nested Data

    private readonly record struct ProjectionInterval(float Minimum, float Maximum);

    private readonly record struct RegionCandidate(
        OcrQuadRegion Region,
        Point2f Point0,
        Point2f Point1,
        Point2f Point2,
        Point2f Point3,
        Point2f Center,
        Point2f Axis,
        float Width,
        float Height,
        float Area)
    {
        public float AspectRatio => Width / Height;
        public float MinX => MathF.Min(MathF.Min(Point0.X, Point1.X), MathF.Min(Point2.X, Point3.X));
        public float MaxX => MathF.Max(MathF.Max(Point0.X, Point1.X), MathF.Max(Point2.X, Point3.X));
        public float MinY => MathF.Min(MathF.Min(Point0.Y, Point1.Y), MathF.Min(Point2.Y, Point3.Y));
        public float MaxY => MathF.Max(MathF.Max(Point0.Y, Point1.Y), MathF.Max(Point2.Y, Point3.Y));
    }

    #endregion
}
