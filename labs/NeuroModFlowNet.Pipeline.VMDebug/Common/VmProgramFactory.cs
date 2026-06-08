using NeuroModFlowNet.Pipeline.ONNX;
using OpenCvSharp;
using CvRect = OpenCvSharp.Rect;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Builds the VM program that is being studied in this lab.
/// </summary>
internal static class VmProgramFactory
{
    /// <summary>
    /// Creates a deliberately small program: Mat image -> OrtValue -> model device memory -> ONNX crop -> ONNX resize -> Mat image.
    /// Optionally includes a direct OBB model run on the resized tensor.
    /// </summary>
    public static VmProgram CreateProgram(string textObbModelPath, bool includeObb)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(textObbModelPath);

        var programBuilder = new VmProgramBuilder();

        // Step 0. Expose CPU OpenCV image memory as an ONNX Runtime tensor.
        // This is zero-copy; the frame Mat must stay alive until the run finishes.
        programBuilder.Step(new Wrap_MatImage_To_OrtTensor(
            inputKey: VmLabKeys.InputImage,
            outputKey: VmLabKeys.InputTensor));

        // Step 1. Ask ONNX Runtime to place the tensor where the model/provider expects it.
        // For CUDA this means device memory. For CPU backend it remains CPU memory, but the VM step stays explicit.
        programBuilder.Step(new Copy_OrtTensor_To_ModelDevice(
            inputKey: VmLabKeys.InputTensor,
            outputKey: VmLabKeys.ModelDeviceTensor));

        // Step 2. Run crop as an ONNX operator, not as OpenCV CPU code.
        // isFinal=false keeps the intermediate tensor in model device placement for the next ONNX operator.
        programBuilder.Step(new Op_Onnx_Crop(
            inputKey: VmLabKeys.ModelDeviceTensor,
            outputKey: VmLabKeys.CroppedTensor,
            cropRect: new CvRect(100, 100, 300, 300),
            isFinal: false));

        // Step 3. Resize the cropped 300x300 tensor to 640x640 to match the expected input of the YOLO model.
        programBuilder.Step(new Op_Onnx_Resize(
            inputKey: VmLabKeys.CroppedTensor,
            outputKey: VmLabKeys.ResizedTensor,
            targetSize: new CvSize(640, 640),
            isFinal: !includeObb));

        if(includeObb)
        {
            // Step 3.5. Run OBB model directly on the resized ONNX tensor, bypassing OpenCV CPU memory.
            programBuilder.Step(new Model_ImgTextToObbRawOrtValue(
                inputKey: VmLabKeys.ResizedTensor,
                outputCountKey: "obb_detections",
                modelPath: textObbModelPath));
        }

        if(!includeObb)
        {
            // Step 4. Convert the resized preview OrtValue into an OpenCV image for visual inspection.
            // This is only possible if the tensor is on the CPU (isFinal: true). If it's on the GPU,
            // copying it directly to a Mat will cause an AccessViolationException.
            programBuilder.Step(new Copy_OrtValue_To_MatImage(
                inputKey: VmLabKeys.ResizedTensor,
                outputKey: VmLabKeys.OutputImage));
        }

        return programBuilder.Build();
    }
}
