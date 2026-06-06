import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "benchmark_runs" / "20260526T114950Z_lstm_compare" / "benchmark_summary.csv"
OUTPUT_PATH = ROOT / "benchmark_model_comparison.png"

MODEL_LABELS = {
    "current_active": "Active",
    "paper_lstm": "Paper LSTM",
    "bilstm": "BiLSTM",
}

MODEL_COLORS = {
    "current_active": "#1f4e79",
    "paper_lstm": "#4f81bd",
    "bilstm": "#9bb9d7",
}

METRICS = [
    ("test_expectancy_r", "Test Expectancy (R)", False),
    ("test_profit_factor", "Test Profit Factor", False),
    ("test_max_drawdown_r", "Test Max Drawdown (R)", True),
    ("test_macro_f1", "Test Macro F1", False),
]


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def load_font(size: int, bold: bool = False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(draw, box, text, font, fill):
    width, height = text_size(draw, text, font)
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2
    draw.text((x, y), text, font=font, fill=fill)


def main():
    rows = load_rows(CSV_PATH)
    ordered_rows = sorted(rows, key=lambda row: list(MODEL_LABELS.keys()).index(row["model_key"]))

    img = Image.new("RGB", (1600, 1100), "white")
    draw = ImageDraw.Draw(img)

    title_font = load_font(34, bold=True)
    subtitle_font = load_font(18)
    panel_title_font = load_font(20, bold=True)
    label_font = load_font(16)
    value_font = load_font(14)
    legend_font = load_font(16)

    draw.text((60, 35), "Benchmark Comparison of LSTM-Based Trading Models", font=title_font, fill="black")
    draw.text(
        (60, 82),
        "Source: benchmark_runs/20260526T114950Z_lstm_compare/benchmark_summary.csv",
        font=subtitle_font,
        fill="#444444",
    )

    legend_x = 980
    legend_y = 38
    for idx, key in enumerate(MODEL_LABELS.keys()):
        y = legend_y + idx * 32
        draw.rectangle((legend_x, y + 4, legend_x + 22, y + 20), fill=MODEL_COLORS[key], outline=MODEL_COLORS[key])
        draw.text((legend_x + 32, y), MODEL_LABELS[key], font=legend_font, fill="black")

    panels = [
        (60, 150, 760, 500),
        (820, 150, 1520, 500),
        (60, 560, 760, 910),
        (820, 560, 1520, 910),
    ]

    for panel, metric in zip(panels, METRICS):
        metric_key, title, invert = metric
        values = [float(row[metric_key]) for row in ordered_rows]
        max_value = max(values)
        min_value = min(values)
        top_value = max_value * 1.15 if max_value > 0 else 1.0
        if invert:
            top_value = max_value * 1.10 if max_value > 0 else 1.0
            min_value = 0.0
        elif max_value == min_value:
            min_value = 0.0

        x0, y0, x1, y1 = panel
        draw.rounded_rectangle(panel, radius=16, outline="#C9D2DA", width=2, fill="#FAFBFC")
        draw.text((x0 + 24, y0 + 18), title, font=panel_title_font, fill="black")

        chart_left = x0 + 58
        chart_right = x1 - 24
        chart_top = y0 + 70
        chart_bottom = y1 - 70
        draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill="#606060", width=2)
        draw.line((chart_left, chart_top, chart_left, chart_bottom), fill="#606060", width=2)

        grid_steps = 4
        for step in range(grid_steps + 1):
            y = chart_bottom - ((chart_bottom - chart_top) * step / grid_steps)
            draw.line((chart_left, y, chart_right, y), fill="#E3E8ED", width=1)
            tick_value = top_value * step / grid_steps
            tick_label = f"{tick_value:.2f}"
            label_w, label_h = text_size(draw, tick_label, value_font)
            draw.text((chart_left - label_w - 10, y - label_h / 2), tick_label, font=value_font, fill="#555555")

        plot_width = chart_right - chart_left
        bar_width = 110
        gap = (plot_width - len(values) * bar_width) / (len(values) + 1)

        for idx, row in enumerate(ordered_rows):
            key = row["model_key"]
            value = float(row[metric_key])
            bar_left = chart_left + gap + idx * (bar_width + gap)
            bar_right = bar_left + bar_width
            ratio = 0.0 if top_value == 0 else value / top_value
            bar_top = chart_bottom - ratio * (chart_bottom - chart_top)
            draw.rectangle((bar_left, bar_top, bar_right, chart_bottom), fill=MODEL_COLORS[key], outline=MODEL_COLORS[key])

            value_label = f"{value:.3f}" if value < 10 else f"{value:.2f}"
            value_w, value_h = text_size(draw, value_label, value_font)
            draw.text((bar_left + (bar_width - value_w) / 2, bar_top - value_h - 8), value_label, font=value_font, fill="black")

            label_box = (bar_left - 8, chart_bottom + 12, bar_right + 8, chart_bottom + 50)
            draw_centered(draw, label_box, MODEL_LABELS[key], label_font, "black")

    caption = "Figure 1. Benchmark comparison across the active architecture, reconstructed paper LSTM, and BiLSTM baseline."
    draw.text((60, 990), caption, font=subtitle_font, fill="#333333")

    img.save(OUTPUT_PATH, format="PNG")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
