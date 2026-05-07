namespace NeuroModFlowNet.ONNX.Visualizer
{
    public struct LetterboxInfo
    {
        public float Ratio;
        public int OffsetX;
        public int OffsetY;
        public int SourceWidth;
        public int SourceHeight;
        public int TargetWidth;
        public int TargetHeight;

        /// <summary>
        /// EN: Maps coordinates from the letterboxed image back to the source image.
        /// <br/>
        /// RU: Пересчитывает координаты из letterboxed изображения обратно в исходное.
        /// </summary>
        public Point2f MapBack(float x, float y)
        {
            return new Point2f((x - OffsetX) / Ratio, (y - OffsetY) / Ratio);
        }

        /// <summary>
        /// EN: Maps scale factor for width/height from letterboxed image back to source image.
        /// <br/>
        /// RU: Пересчитывает коэффициент масштабирования длины/ширины из letterboxed изображения обратно в исходное.
        /// </summary>
        public float MapScale(float value) => value / Ratio;
    }
}
