$DownloadConvertImgTextToObb = Join-Path $PSScriptRoot "DownloadConvertImgTextToObb.ps1"

& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr false
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr true
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr false
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1   -Half false -Dynamic true  -ByteBgr true
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr false
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1,4 -Half true  -Dynamic false -ByteBgr true
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr false
& $DownloadConvertImgTextToObb -Models img-text-to-obb  -ImgSize 640 -Batches 1   -Half true  -Dynamic true  -ByteBgr true
