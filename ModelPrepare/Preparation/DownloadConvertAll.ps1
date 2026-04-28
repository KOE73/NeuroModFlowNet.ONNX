$DownloadConvertYolo         = Join-Path $PSScriptRoot "DownloadConvertYolo.ps1"
$DownloadConvertPadle        = Join-Path $PSScriptRoot "DownloadConvertPadle.ps1"
$DownloadConvertImgTextToObb = Join-Path $PSScriptRoot "DownloadConvertImgTextToObb.ps1"

$YoloModel     = "yolo26n"
$YoloObbModel  = "$YoloModel-obb"
$YoloSegModel  = "$YoloModel-seg"
$YoloPoseModel = "$YoloModel-pose"
$YoloClsModel  = "$YoloModel-cls"

& $DownloadConvertPadle -Url "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v3/det.onnx" -OutputPath "paddleocr/detection/v3/det.onnx" -InjectHead false
& $DownloadConvertPadle -Url "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v3/det.onnx" -OutputPath "paddleocr/detection/v3/det.onnx" -InjectHead true
& $DownloadConvertPadle -Url "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v5/det.onnx" -OutputPath "paddleocr/detection/v5/det.onnx" -InjectHead false
& $DownloadConvertPadle -Url "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v5/det.onnx" -OutputPath "paddleocr/detection/v5/det.onnx" -InjectHead true

& $DownloadConvertYolo -Models $YoloModel     -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr false
& $DownloadConvertYolo -Models $YoloModel     -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr true
& $DownloadConvertYolo -Models $YoloModel     -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr false
& $DownloadConvertYolo -Models $YoloModel     -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr true
& $DownloadConvertYolo -Models $YoloModel     -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr false
& $DownloadConvertYolo -Models $YoloModel     -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr true
& $DownloadConvertYolo -Models $YoloModel     -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr false
& $DownloadConvertYolo -Models $YoloModel     -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr true

& $DownloadConvertYolo -Models $YoloObbModel  -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr false
& $DownloadConvertYolo -Models $YoloObbModel  -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr true
& $DownloadConvertYolo -Models $YoloObbModel  -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr false
& $DownloadConvertYolo -Models $YoloObbModel  -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr true
& $DownloadConvertYolo -Models $YoloObbModel  -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr false
& $DownloadConvertYolo -Models $YoloObbModel  -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr true
& $DownloadConvertYolo -Models $YoloObbModel  -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr false
& $DownloadConvertYolo -Models $YoloObbModel  -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr true

& $DownloadConvertYolo -Models $YoloSegModel  -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr false
& $DownloadConvertYolo -Models $YoloSegModel  -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr true
& $DownloadConvertYolo -Models $YoloSegModel  -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr false
& $DownloadConvertYolo -Models $YoloSegModel  -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr true
& $DownloadConvertYolo -Models $YoloSegModel  -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr false
& $DownloadConvertYolo -Models $YoloSegModel  -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr true
& $DownloadConvertYolo -Models $YoloSegModel  -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr false
& $DownloadConvertYolo -Models $YoloSegModel  -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr true

& $DownloadConvertYolo -Models $YoloPoseModel -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr false
& $DownloadConvertYolo -Models $YoloPoseModel -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr true
& $DownloadConvertYolo -Models $YoloPoseModel -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr false
& $DownloadConvertYolo -Models $YoloPoseModel -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr true
& $DownloadConvertYolo -Models $YoloPoseModel -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr false
& $DownloadConvertYolo -Models $YoloPoseModel -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr true
& $DownloadConvertYolo -Models $YoloPoseModel -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr false
& $DownloadConvertYolo -Models $YoloPoseModel -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr true

& $DownloadConvertYolo -Models $YoloClsModel  -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr false
& $DownloadConvertYolo -Models $YoloClsModel  -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr true
& $DownloadConvertYolo -Models $YoloClsModel  -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr false
& $DownloadConvertYolo -Models $YoloClsModel  -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr true
& $DownloadConvertYolo -Models $YoloClsModel  -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr false
& $DownloadConvertYolo -Models $YoloClsModel  -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr true
& $DownloadConvertYolo -Models $YoloClsModel  -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr false
& $DownloadConvertYolo -Models $YoloClsModel  -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr true

& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr false
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr true
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr false
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr true
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr false
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr true
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr false
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr true
