"""Render documentation data-flow diagrams as editable SVG and high-resolution PNG.

The JSON specification is documentation artwork. SeMaps receives its entities,
relations and placements through MCP; this script never reads or writes a SeMaps workspace.
"""

from pathlib import Path
import html
import json
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PALETTE = {
    "gpu": ("#eff6ff", "#3b82f6"),
    "cpu": ("#fff7ed", "#ea580c"),
    "boundary": ("#f0fdf4", "#16a34a"),
}


def node(key, ru, en, x, y, role, description, english, width=440):
    return dict(key=key, ru=ru, en=en, x=x, y=y, width=width, height=100,
                role=role, description=description, english=english)


def edge(start, end, label, route=None, label_pos=None):
    result = dict(start=start, end=end, label=label)
    if route:
        result["route"] = route
    if label_pos:
        result["labelPos"] = label_pos
    return result


def build_specification():
    ingest = dict(id="v_video_ingest_dataflow", slug="video-ingest", width=1420, height=1480,
                  ru="Видео → рабочий кадр: потоки данных", en="Video to working frame: data flow", nodes=[], edges=[])
    ingest["nodes"] = [
        node("source", "Видеофайл / RTSP", "Video file / RTSP", 490, 140, "boundary", "Сжатое видео. NVDEC принимает файлы и RTSP; USB-источник доступен через OpenCV.", "Compressed video. NVDEC accepts files and RTSP; USB capture is available through OpenCV."),
        node("nvdec", "FFmpeg + NVDEC", "FFmpeg + NVDEC", 100, 380, "gpu", "NvdecVideoDecoder: demux и аппаратный декод в CUDA NV12. Без CPU fallback.", "NvdecVideoDecoder: demux and hardware decode into CUDA NV12, without CPU fallback."),
        node("opencv", "OpenCV: декод в Mat", "OpenCV: decode to Mat", 880, 380, "cpu", "Явно выбранный CPU/Mat путь; также поддерживает USB-камеру.", "Explicit CPU/Mat input path; also supports USB camera capture."),
        node("ort_nv12", "OrtValue над NV12", "OrtValue over NV12", 100, 620, "gpu", "NvdecFrame.CreateNv12Tensor: zero-copy view [allocatedHeight*3/2, pitch] U8. Поверхность удерживается до завершения run; CudaDecodeFence обеспечивает готовность данных.", "NvdecFrame.CreateNv12Tensor: zero-copy view [allocatedHeight*3/2, pitch] U8. The surface remains alive until the run completes; CudaDecodeFence ensures data readiness."),
        node("ort_cpu", "OrtValue над Mat", "OrtValue over Mat", 880, 620, "cpu", "Wrap_MatImage_To_OrtTensor: обёртка над пикселями CPU Mat.", "Wrap_MatImage_To_OrtTensor wraps the CPU Mat pixels."),
        node("convert", "ONNX: NV12 → BGR", "ONNX: NV12 to BGR", 100, 860, "gpu", "Op_Onnx_Nv12_To_BgrU8Nhwc: явная цветовая матрица BT.601/BT.709 и диапазон; создаёт новый BGR tensor на GPU.", "Op_Onnx_Nv12_To_BgrU8Nhwc uses an explicit BT.601/BT.709 matrix and range; it creates a new BGR tensor on the GPU."),
        node("upload", "Явный upload", "Explicit upload", 880, 860, "boundary", "Copy_OrtTensor_To_ModelDevice: явный перенос пикселей с CPU на устройство модели.", "Copy_OrtTensor_To_ModelDevice explicitly transfers pixels from CPU to the model device."),
        node("prepare", "Геометрия камеры", "Camera geometry", 490, 1100, "gpu", "CameraPrepareProgramFactory: configured undistort, perspective, rotate, crop, resize в заданном порядке.", "CameraPrepareProgramFactory: configured undistort, perspective, rotate, crop and resize in the declared order."),
        node("rect", "image.rect", "image.rect", 490, 1340, "gpu", "Рабочий кадр BGR U8 NHWC [1,H,W,3] после геометрии; единое координатное пространство common.", "Working BGR U8 NHWC [1,H,W,3] frame after geometry; the common program uses this coordinate space."),
    ]
    ingest["edges"] = [
        edge("source", "nvdec", "Сжатые видеопакеты"), edge("source", "opencv", "Видеоданные"),
        edge("nvdec", "ort_nv12", "CUDA NV12-поверхность"), edge("opencv", "ort_cpu", "CPU Mat: BGR U8"),
        edge("ort_nv12", "convert", "GPU OrtValue: NV12 U8"), edge("ort_cpu", "upload", "CPU OrtValue: BGR U8 NHWC"),
        edge("convert", "prepare", "GPU BGR U8 NHWC"), edge("upload", "prepare", "Device BGR U8 NHWC"),
        edge("prepare", "rect", "OrtValue [1,H,W,3]"),
    ]
    common = dict(id="v_multicamera_dataflow", slug="multicamera", width=1840, height=2410,
                  ru="MultiCamera: потоки данных", en="MultiCamera: data flow", nodes=[], edges=[])
    common["nodes"] = [
        node("rect", "image.rect", "image.rect", 680, 140, "gpu", "Полноразмерный рабочий кадр после prepare. Все координаты ниже приводятся в rect space.", "Full-resolution working frame after prepare. All coordinates below are mapped to rect space."),
        node("det_prepare", "Вход детектора мешков", "Bag detector input", 120, 380, "gpu", "Resize/letterbox по размеру модели камеры; BGR U8 → RGB FP16/FP32 NCHW /255. Также выдаёт back-transform.", "Resize/letterbox to the camera model size; BGR U8 to RGB FP16/FP32 NCHW /255. Also outputs a back-transform."),
        node("rgb", "Полноразмерный RGB", "Full-resolution RGB", 1240, 380, "gpu", "RGB FP32 NCHW 0..1 из image.rect, источник пикселей для вырезок мешков и текстовых ROI при включённом OCR.", "RGB FP32 NCHW 0..1 from image.rect; the pixel source for bag crops and text ROIs when OCR is enabled."),
        node("detector", "Детектор мешков", "Bag detector", 120, 620, "gpu", "Model_OrtValueInference<YoloBox/YoloObb>; endpoint на файл модели. MaxBatchSize=1, MaxWaitTime=0.", "Model_OrtValueInference<YoloBox/YoloObb>; one endpoint per model file. MaxBatchSize=1, MaxWaitTime=0."),
        node("preview_resize", "Уменьшение превью", "Preview resize", 680, 620, "gpu", "Опциональная ветка: сначала уменьшить кадр на GPU; isFinal=true размещает выход в CPU memory.", "Optional branch: resize on the GPU first; isFinal=true places the output in CPU memory."),
        node("det_map", "Детекции → rect", "Detections to rect", 120, 860, "cpu", "Op_Map_Coordinates и отбор класса преобразуют готовые typed boxes в TrackDetection[].",
             "Op_Map_Coordinates and class filtering turn typed boxes into TrackDetection[]."),
        node("preview_mat", "Mat: превью с подписями", "Mat: annotated preview", 680, 860, "cpu", "Copy_OrtValue_To_MatImage; CPU-отрисовка получает готовые detections/tracks/OCR results. Опционально окно и PNG snapshots.", "Copy_OrtValue_To_MatImage; CPU rendering consumes detections, tracks and OCR results. Optional window and PNG snapshots."),
        node("tracker", "Трекер камеры", "Camera tracker", 120, 1100, "cpu", "Op_Track: состояние в VmGlobalMemory камеры, обновление в ordered critical section; выход TrackedObject[].",
             "Op_Track: state in the camera VmGlobalMemory, updated in an ordered critical section; output TrackedObject[]."),
        node("select", "Области мешков + ID", "Bag regions + track IDs", 120, 1340, "cpu", "Op_TracksToCropBoxes: подтверждённые треки с MissedFrames=0, не более maxCrops; YoloObb[] и int[].",
             "Op_TracksToCropBoxes: confirmed tracks with MissedFrames=0, at most maxCrops; YoloObb[] and int[]."),
        node("crops", "Вырезки мешков", "Bag crops", 680, 1340, "gpu", "Op_Onnx_ExtractObbToLetterboxBatch_FP32_NCHW: [maxCrops,3,S,S], transforms[] и actualCount. Явная политика Truncate.", "Op_Onnx_ExtractObbToLetterboxBatch_FP32_NCHW: [maxCrops,3,S,S], transforms[] and actualCount. Explicit Truncate policy."),
        node("text_detector", "Детектор текстовых OBB", "Text OBB detector", 400, 1580, "gpu", "Общий endpoint принимает готовый batch вырезок. MaxBatchSize=1; CropTextRegion[] сохраняет crop index.", "Shared endpoint accepts a prepared crop batch. MaxBatchSize=1; CropTextRegion[] retains the crop index."),
        node("text_map", "Текстовые области → rect", "Text regions to rect", 680, 1820, "cpu", "Op_MapCropTextRegionsToRect: OBB обратно в rect и параллельные track IDs; результаты padding crop slots отбрасываются.", "Op_MapCropTextRegionsToRect: OBBs back to rect plus parallel track IDs; padding crop slot results are discarded."),
        node("roi", "Подготовка текстовых ROI", "Text ROI preparation", 1240, 1820, "gpu", "Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW: повторная вырезка из полноразмерного RGB, геометрия и Paddle-нормализация; actualCount отдельно.", "Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW: re-extract from full-resolution RGB, geometry and Paddle normalization; separate actualCount."),
        node("recognition", "PaddleOCR Rec", "PaddleOCR Rec", 1240, 2060, "gpu", "OrtValue endpoint: [N,3,H,W] → OcrResult[]. Backend recognition задан явно; скрытый fallback запрещён. MaxBatchSize=1.", "OrtValue endpoint: [N,3,H,W] to OcrResult[]. Recognition backend is explicit; hidden fallback is forbidden. MaxBatchSize=1."),
        node("results", "Текст + ID треков", "Text + track IDs", 1240, 2300, "cpu", "CameraWorker отсекает padding recognition slots по actualCount, связывает строки с track IDs; результат и статистика.", "CameraWorker trims padding recognition slots by actualCount and associates strings with track IDs; results and statistics."),
    ]
    common["edges"] = [
        edge("rect", "det_prepare", "BGR U8 NHWC"), edge("rect", "rgb", "BGR U8 NHWC"),
        edge("det_prepare", "detector", "RGB FP16/FP32 NCHW"),
        edge("det_prepare", "det_map", "Back-transform", [[120,430],[60,430],[60,910],[120,910]], [160,750]),
        edge("detector", "det_map", "YoloBox[] / YoloObb[]"), edge("det_map", "tracker", "TrackDetection[]"),
        edge("tracker", "select", "TrackedObject[]"), edge("select", "crops", "OBB[]", label_pos=[620,1385]),
        edge("rgb", "crops", "RGB FP32: пиксели", [[1240,430],[1160,430],[1160,1390],[1120,1390]], [1160,1070]),
        edge("crops", "text_detector", "GPU batch [maxCrops,3,S,S]"),
        edge("crops", "text_map", "Transforms[] / count", [[1120,1390],[1190,1390],[1190,1780],[1100,1780],[1100,1820]], [1190,1730]),
        edge("select", "text_map", "ID треков", [[340,1440],[340,1870],[680,1870]], [340,1740]),
        edge("text_detector", "text_map", "CropTextRegion[]"), edge("text_map", "roi", "OBB rect", label_pos=[1180,1865]),
        edge("rgb", "roi", "Полноразмерные пиксели", [[1680,430],[1760,430],[1760,1870],[1680,1870]], [1600,1240]),
        edge("roi", "recognition", "FP32 NCHW + actualCount"), edge("recognition", "results", "OcrResult[]"),
        edge("text_map", "results", "ID треков", [[1120,1870],[1180,1870],[1180,2350],[1240,2350]], [1180,2190]),
        edge("rect", "preview_resize", "BGR U8 NHWC"), edge("preview_resize", "preview_mat", "CPU tensor → Mat"),
    ]
    videoort = dict(id="v_videoort_dataflow", slug="videoort", width=1820, height=1720,
                    ru="VideoOrt: verify / bench, потоки данных", en="VideoOrt: verify / bench data flow", nodes=[], edges=[])
    videoort["nodes"] = [
        node("source", "Исходное видео", "Source video", 680, 140, "boundary", "Verify: один файл. Bench: файлы/RTSP, N независимых streams.", "Verify: one file. Bench: files/RTSP and N independent streams."),
        node("nvdec", "FFmpeg + NVDEC", "FFmpeg + NVDEC", 120, 380, "gpu", "Собственный NvdecVideoDecoder на stream. UseFence=true по умолчанию; --decode-only измеряет только этот участок.", "One NvdecVideoDecoder per stream. UseFence=true by default; --decode-only measures only this section."),
        node("cpu", "OpenCV: CPU-эталон", "OpenCV: CPU reference", 1240, 380, "cpu", "VerifyCommand декодирует тот же файл по кадрам через OpenCV.", "VerifyCommand decodes the same file frame by frame through OpenCV."),
        node("ort", "OrtValue над NV12", "OrtValue over NV12", 120, 620, "gpu", "Zero-copy обёртка поверхности CUDA с удержанием NvdecFrame до конца VM run.", "Zero-copy wrapper over the CUDA surface; NvdecFrame remains alive until the VM run ends."),
        node("convert", "ONNX: NV12 → BGR", "ONNX: NV12 to BGR", 120, 860, "gpu", "BGR U8 NHWC. Verify: isFinal=true, выход CPU. Bench: isFinal=false, выход GPU.", "BGR U8 NHWC. Verify: isFinal=true, CPU output. Bench: isFinal=false, GPU output."),
        node("download", "Mat после GPU-пути", "Mat from GPU path", 120, 1100, "cpu", "Verify: Copy_OrtValue_To_MatImage; намеренная выгрузка для сравнения пикселей.", "Verify: Copy_OrtValue_To_MatImage; intentional download for pixel comparison."),
        node("resize", "Bench: GPU resize", "Bench: GPU resize", 680, 860, "gpu", "Op_Onnx_Resize_U8_NHWC, без выгрузки кадра; один VmController на stream.", "Op_Onnx_Resize_U8_NHWC, without frame download; one VmController per stream."),
        node("statistics", "Bench: измерения", "Bench: measurements", 680, 1100, "cpu", "Completed frames, decode ms, VM ms, total/per-stream FPS. Это скорость видеотракта без детекции, трекера и OCR.", "Completed frames, decode ms, VM ms, total/per-stream FPS. This measures the video path without detection, tracking or OCR."),
        node("compare", "Verify: сравнение пикселей", "Verify: pixel comparison", 680, 1340, "cpu", "Cv2.Absdiff; средняя и максимальная абсолютная разница, покадровое соответствие.", "Cv2.Absdiff; mean and maximum absolute difference with frame-by-frame matching."),
        node("artifacts", "Verify: PNG + метрики", "Verify: PNG + metrics", 680, 1580, "cpu", "Первый/худший кадр: GPU-путь, CPU-эталон, diff*8. Pass: compared>0 и средняя разница ниже threshold; побитовая идентичность не требуется.", "First/worst frame: GPU path, CPU reference, diff*8. Pass: compared>0 and average difference below threshold; bitwise identity is not required."),
    ]
    videoort["edges"] = [
        edge("source", "nvdec", "Сжатые видеопакеты"), edge("source", "cpu", "Тот же файл"),
        edge("nvdec", "ort", "CUDA NV12"), edge("ort", "convert", "GPU OrtValue"),
        edge("convert", "download", "Verify: CPU BGR tensor"), edge("convert", "resize", "GPU BGR"),
        edge("resize", "statistics", "Число кадров / время"),
        edge("download", "compare", "BGR Mat после GPU"),
        edge("cpu", "compare", "CPU reference Mat", [[1460,480],[1460,1390],[1120,1390]], [1460,1010]),
        edge("compare", "artifacts", "Diff / mean / max"),
    ]
    return [ingest, common, videoort]


def automatic_route(first, second):
    a = (first["x"] + first["width"]/2, first["y"] + first["height"]/2)
    b = (second["x"] + second["width"]/2, second["y"] + second["height"]/2)
    if abs(a[1]-b[1]) < 1:
        direction = 1 if b[0] > a[0] else -1
        return [(a[0] + direction*first["width"]/2, a[1]), (b[0] - direction*second["width"]/2, b[1])]
    start = (a[0], first["y"] + first["height"])
    end = (b[0], second["y"])
    mid = (start[1]+end[1])/2
    return [start, (start[0],mid), (end[0],mid), end]


def render(specification):
    scale = 2
    width, height = specification["width"], specification["height"]
    image = Image.new("RGB", (width*scale, height*scale), "#ffffff")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22*scale)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 30*scale)
    node_font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 24*scale)
    small_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18*scale)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/></marker></defs>']

    def text_at(value, point, chosen_font, size, color="#0f172a", bold=False):
        escaped = html.escape(value)
        draw.text((point[0]*scale, point[1]*scale), value, font=chosen_font, fill=color, anchor="mm")
        svg.append(f'<text x="{point[0]}" y="{point[1]}" text-anchor="middle" dominant-baseline="central" font-family="Segoe UI, Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escaped}</text>')

    text_at(specification["ru"], (width/2,42), title_font, 30, bold=True)
    text_at("Только потоки данных • синий: GPU • оранжевый: CPU • зелёный: граница", (width/2,84), small_font, 18, "#475569")
    lookup = {item["key"]: item for item in specification["nodes"]}
    labels = []
    for relation in specification["edges"]:
        route = relation.get("route") or automatic_route(lookup[relation["start"]], lookup[relation["end"]])
        draw.line([(x*scale,y*scale) for x,y in route], fill="#2563eb", width=3*scale, joint="curve")
        p, q = route[-2:]
        angle = math.atan2(q[1]-p[1],q[0]-p[0])
        corners = [(q[0],q[1]), (q[0]-14*math.cos(angle-0.45),q[1]-14*math.sin(angle-0.45)), (q[0]-14*math.cos(angle+0.45),q[1]-14*math.sin(angle+0.45))]
        draw.polygon([(x*scale,y*scale) for x,y in corners], fill="#2563eb")
        points = " ".join(f"{x},{y}" for x,y in route)
        svg.append(f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3" marker-end="url(#arrow)"/>')
        position = relation.get("labelPos")
        if not position:
            first, last = route[0],route[-1]
            position = [(first[0]+last[0])/2, (first[1]+last[1])/2-12]
        labels.append((relation["label"], position))

    for item in specification["nodes"]:
        fill, border = PALETTE[item["role"]]
        x,y,w,h = item["x"],item["y"],item["width"],item["height"]
        draw.rounded_rectangle((x*scale,y*scale,(x+w)*scale,(y+h)*scale), radius=12*scale, fill=fill, outline=border,width=2*scale)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{border}" stroke-width="2"><title>{html.escape(item["description"])}</title></rect>')
        text_at(item["ru"], (x+w/2,y+h/2-10), node_font,24,bold=True)
        role = {"gpu":"GPU / tensor", "cpu":"CPU / managed data", "boundary":"Вход / граница памяти"}[item["role"]]
        text_at(role,(x+w/2,y+h/2+24),small_font,18,"#475569")

    for label, position in labels:
        box = draw.textbbox((0,0),label,font=font)
        label_width = (box[2]-box[0])/scale+16
        x,y=position
        x = max(label_width/2 + 12, min(width-label_width/2-12, x))
        draw.rectangle(((x-label_width/2)*scale,(y-16)*scale,(x+label_width/2)*scale,(y+16)*scale),fill="white")
        svg.append(f'<rect x="{x-label_width/2}" y="{y-16}" width="{label_width}" height="32" rx="4" fill="white"/>')
        text_at(label, position, font,22,"#1d4ed8")
    svg.append('</svg>')
    (ROOT/f'{specification["slug"]}.svg').write_text("\n".join(svg),encoding="utf-8")
    image.save(ROOT/f'{specification["slug"]}.png')


if __name__ == "__main__":
    diagrams = build_specification()
    (ROOT/"dataflow-spec.json").write_text(json.dumps(diagrams,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for diagram in diagrams:
        render(diagram)
        print(f'{diagram["slug"]}: {diagram["width"]*2}x{diagram["height"]*2}')
