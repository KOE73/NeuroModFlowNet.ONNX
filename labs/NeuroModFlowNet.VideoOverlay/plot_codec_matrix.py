"""Plot the measured D3D11VA / Direct2D / NVENC benchmark grid."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def read_matrix(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    inputs = sorted({int(row["inputs"]) for row in rows})
    outputs = sorted({int(row["encoders"]) for row in rows})
    pairs = {(int(row["inputs"]), int(row["encoders"])): row for row in rows}
    if not rows or len(rows) != len(pairs) or len(pairs) != len(inputs) * len(outputs):
        raise ValueError("expected one measurement per input/encoder pair")
    return np.meshgrid(inputs, outputs), pairs


def plot_surface(axis, grid, pairs, key, title, unit, palette, scale):
    inputs, outputs = grid
    values = np.full(inputs.shape, np.nan)
    for index in np.ndindex(inputs.shape):
        raw = pairs[inputs[index], outputs[index]][key]
        if raw:
            values[index] = float(raw) * scale

    axis.plot_surface(inputs, outputs, values, cmap=palette, rstride=1, cstride=1,
                      linewidth=0.22, edgecolor=(0.04, 0.06, 0.12, 0.20), antialiased=True)
    axis.set_title(title, fontsize=14, fontweight="bold", pad=4)
    axis.set_xlabel("Входы", labelpad=9)
    axis.set_ylabel("Сессии NVENC", labelpad=8)
    axis.set_zlabel(unit, labelpad=9)
    axis.set_xticks([value for value in (1, 10, 20, 30, 40, 50)
                     if inputs.min() <= value <= inputs.max()])
    axis.set_yticks([value for value in range(0, 11, 2) if outputs.min() <= value <= outputs.max()])
    if key.endswith("fps"):
        axis.zaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", " ")))
    axis.set_box_aspect((1.35, 1.0, 0.78))
    axis.view_init(elev=28, azim=-58)
    for direction in (axis.xaxis, axis.yaxis, axis.zaxis):
        direction.pane.set_facecolor("#edf1f5")
        direction.pane.set_edgecolor("#9ca9b8")


def make_figure(grid, pairs, panels, title, subtitle, destination, duration_ms):
    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": "#13233a"}):
        figure = plt.figure(figsize=(14.8, 7.1), facecolor="white")
        figure.text(0.052, 0.946, title, fontsize=19, fontweight="bold")
        figure.text(0.052, 0.906, subtitle, fontsize=11, color="#455970")
        for index, panel in enumerate(panels, start=1):
            axis = figure.add_subplot(1, 2, index, projection="3d")
            plot_surface(axis, grid, pairs, *panel)
        figure.subplots_adjust(left=0.025, right=0.955, bottom=0.08, top=0.86, wspace=0.02)
        figure.text(0.052, 0.031, f"RTX 5090  ·  1920×1080  ·  25 кадров/с на вход  ·  "
                    f"{len(pairs)} точек  ·  окно {duration_ms:.0f} мс на точку",
                    fontsize=10, color="#455970")
        figure.savefig(destination, dpi=155)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    grid, pairs = read_matrix(args.csv_path)
    first = next(iter(pairs.values()))
    duration_ms = 1000 * float(first["offered_frames"]) / float(first["offered_fps"])
    args.output_directory.mkdir(parents=True, exist_ok=True)
    make_figure(
        grid, pairs,
        [
            ("processed_fps", "Все входы после рисования и конвертаций", "кадров/с", "viridis", 1.0),
            ("encoded_fps", "Пакеты H.264 из выбранных NVENC-сессий", "пакетов/с", "cividis", 1.0),
        ],
        "Как число декодеров и энкодеров меняет пропускную способность",
        "Поверхности построены по измеренной сетке. При нескольких выходах на один вход кадр рисуется один раз.",
        args.output_directory / "codec-fps-3d.png", duration_ms,
    )
    make_figure(
        grid, pairs,
        [
            ("process_p95_ms", "До готовой NV12-текстуры", "p95, с", "plasma", 0.001),
            ("encode_p95_ms", "До получения H.264 в RAM", "p95, с", "magma", 0.001),
        ],
        "Задержка кадра под нагрузкой",
        "От запланированной подачи пакета, включая очередь. Поверхность H.264 начинается с одной сессии.",
        args.output_directory / "codec-latency-3d.png", duration_ms,
    )


if __name__ == "__main__":
    main()
