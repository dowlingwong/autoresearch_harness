#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export KDD AAE paper SVG figures.")
    parser.add_argument(
        "--figure",
        required=True,
        choices=("architecture", "repeated_bad_rate", "decision_breakdown", "trajectory"),
    )
    parser.add_argument("--input", help="Input CSV for data-backed figures.")
    parser.add_argument("--output", required=True, help="Output SVG path.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.figure == "architecture":
        svg = _architecture_svg()
    else:
        if not args.input:
            parser.error("--input is required for this figure")
        rows = _read_csv(Path(args.input))
        if args.figure == "repeated_bad_rate":
            svg = _repeated_bad_rate_svg(rows)
        elif args.figure == "decision_breakdown":
            svg = _decision_breakdown_svg(rows)
        else:
            svg = _trajectory_svg(rows)
    output.write_text(svg, encoding="utf-8")
    print(output)
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        "<style>"
        "text{font-family:Arial,sans-serif;fill:#1b1b1b}"
        ".title{font-size:20px;font-weight:700}.label{font-size:12px}.small{font-size:10px}"
        ".axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}"
        "</style>\n"
        f"{body}\n</svg>\n"
    )


def _repeated_bad_rate_svg(rows: list[dict[str, str]]) -> str:
    width, height = 760, 440
    left, top, chart_w, chart_h = 80, 70, 620, 280
    values = [(row["memory_mode"], _float(row.get("repeated_bad_rate"))) for row in rows]
    max_y = max([value for _, value in values] + [0.1])
    bar_w = chart_w / max(len(values), 1) * 0.55
    parts = [
        '<text x="80" y="34" class="title">Figure 2. Repeated-bad rate by memory mode</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" class="axis"/>',
    ]
    for i in range(5):
        y = top + chart_h - (chart_h * i / 4)
        value = max_y * i / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="42" y="{y + 4:.1f}" class="small">{value:.2f}</text>')
    colors = ["#2f6f8f", "#4b9b6f", "#c77d35"]
    for idx, (mode, value) in enumerate(values):
        x_center = left + (idx + 0.5) * chart_w / max(len(values), 1)
        h = 0 if max_y == 0 else chart_h * value / max_y
        x = x_center - bar_w / 2
        y = top + chart_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[idx % len(colors)]}"/>')
        parts.append(f'<text x="{x_center:.1f}" y="{top + chart_h + 22}" text-anchor="middle" class="small">{_short_mode(mode)}</text>')
        parts.append(f'<text x="{x_center:.1f}" y="{max(y - 8, top + 12):.1f}" text-anchor="middle" class="label">{value:.2f}</text>')
    return _svg(width, height, "\n".join(parts))


def _decision_breakdown_svg(rows: list[dict[str, str]]) -> str:
    width, height = 860, 460
    left, top, chart_w, chart_h = 100, 70, 690, 300
    parts = [
        '<text x="80" y="34" class="title">Figure 3. Decision breakdown by campaign</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" class="axis"/>',
    ]
    colors = {"kept": "#4b9b6f", "discarded": "#d8a13f", "failed_invalid": "#b85450"}
    bar_h = min(34, chart_h / max(len(rows), 1) * 0.55)
    for idx, row in enumerate(rows):
        total = max(_int(row.get("total_trials")), 1)
        y = top + 24 + idx * chart_h / max(len(rows), 1)
        x = left
        for key in ("kept", "discarded", "failed_invalid"):
            value = _int(row.get(key))
            w = chart_w * value / total
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="{colors[key]}"/>')
            if value:
                parts.append(f'<text x="{x + w / 2:.1f}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="middle" class="small">{value}</text>')
            x += w
        parts.append(f'<text x="{left - 8}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end" class="small">{html.escape(row.get("campaign_id", ""))}</text>')
    legend_x = left
    for idx, key in enumerate(("kept", "discarded", "failed_invalid")):
        x = legend_x + idx * 150
        parts.append(f'<rect x="{x}" y="400" width="14" height="14" fill="{colors[key]}"/>')
        parts.append(f'<text x="{x + 20}" y="412" class="label">{key}</text>')
    return _svg(width, height, "\n".join(parts))


def _trajectory_svg(rows: list[dict[str, str]]) -> str:
    width, height = 820, 500
    left, top, chart_w, chart_h = 100, 70, 640, 300
    bottom = top + chart_h

    # Parse all rows (including failed trials with no metric)
    all_trials = [
        {
            "idx": _int(row.get("budget_index")),
            "metric": _float(row.get("metric_value")) if row.get("metric_value") else None,
            "decision": row.get("decision", ""),
        }
        for row in rows
    ]
    valid_points = [(t["idx"], t["metric"]) for t in all_trials if t["metric"] is not None]
    ys = [m for _, m in valid_points] or [0.0]
    min_y, max_y = min(ys), max(ys)
    y_pad = max((max_y - min_y) * 0.15, 0.0005)
    plot_min = min_y - y_pad
    plot_max = max_y + y_pad
    max_x = max([t["idx"] for t in all_trials] + [1])

    def x_pos(idx: int) -> float:
        return left + chart_w * (idx - 1) / max(max_x - 1, 1)

    def y_pos(val: float) -> float:
        return bottom - chart_h * (val - plot_min) / (plot_max - plot_min)

    decision_color = {"kept": "#4b9b6f", "discarded": "#d8a13f", "failed_invalid": "#b85450"}

    parts: list[str] = []

    # Title
    parts.append('<text x="100" y="34" class="title">Figure 4. Campaign metric trajectory</text>')

    # Y-axis gridlines + tick labels (5 evenly spaced)
    n_ticks = 5
    for i in range(n_ticks + 1):
        val = plot_min + (plot_max - plot_min) * i / n_ticks
        y = y_pos(val)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{val:.4f}</text>')

    # Axes
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{left + chart_w}" y2="{bottom}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>  ')

    # X-axis tick labels + vertical reference lines for each trial
    for trial in all_trials:
        x = x_pos(trial["idx"])
        parts.append(f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 5}" stroke="#333" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" class="small">T{trial["idx"]}</text>')
        # Dashed drop-line for failed trials so they're visible despite no data point
        if trial["decision"] == "failed_invalid":
            fail_color = decision_color["failed_invalid"]
            parts.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
                f'stroke="{fail_color}" stroke-width="1" stroke-dasharray="4 3" opacity="0.5"/>'
            )
            # X marker at mid-chart height
            mx, my = x, (top + bottom) / 2
            r = 7
            parts.append(f'<line x1="{mx - r}" y1="{my - r}" x2="{mx + r}" y2="{my + r}" stroke="{fail_color}" stroke-width="2"/>')
            parts.append(f'<line x1="{mx - r}" y1="{my + r}" x2="{mx + r}" y2="{my - r}" stroke="{fail_color}" stroke-width="2"/>')

    # Axis labels
    parts.append(f'<text x="{left + chart_w / 2}" y="{bottom + 40}" text-anchor="middle" class="label">Trial</text>')
    parts.append(
        f'<text x="18" y="{top + chart_h / 2}" text-anchor="middle" class="label" '
        f'transform="rotate(-90,18,{top + chart_h / 2:.1f})">val_AUC</text>'
    )

    # Trajectory line through valid points only
    if valid_points:
        px_coords = [(x_pos(idx), y_pos(m)) for idx, m in valid_points]
        d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(px_coords))
        parts.append(f'<path d="{d}" fill="none" stroke="#2f6f8f" stroke-width="2" stroke-dasharray="5 3"/>')

    # Decision-coloured data points on top
    for trial in all_trials:
        if trial["metric"] is None:
            continue
        x = x_pos(trial["idx"])
        y = y_pos(trial["metric"])
        color = decision_color.get(trial["decision"], "#888")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{y - 11:.1f}" text-anchor="middle" class="small">{trial["metric"]:.4f}</text>')

    # Legend
    legend_items = [("kept", "#4b9b6f"), ("discarded", "#d8a13f"), ("failed_invalid", "#b85450")]
    leg_y = bottom + 60
    for i, (label, color) in enumerate(legend_items):
        lx = left + i * 190
        parts.append(f'<circle cx="{lx + 7}" cy="{leg_y}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{leg_y + 4}" class="small">{label}</text>')

    return _svg(width, height, "\n".join(parts))


def _architecture_svg() -> str:
    width, height = 880, 420
    boxes = [
        ("Manager", 50, 70),
        ("Control Plane", 250, 70),
        ("Memory / Ledger", 250, 230),
        ("Worker", 480, 70),
        ("Training", 650, 70),
        ("Metric Parser", 650, 220),
        ("Decision", 480, 220),
    ]
    parts = ['<text x="50" y="34" class="title">Figure 1. Governed autonomous experimentation architecture</text>']
    for label, x, y in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="150" height="58" rx="6" fill="#f6f7f8" stroke="#333"/>')
        parts.append(f'<text x="{x + 75}" y="{y + 35}" text-anchor="middle" class="label">{label}</text>')
    arrows = [
        (200, 99, 250, 99),
        (400, 99, 480, 99),
        (630, 99, 650, 99),
        (725, 128, 725, 220),
        (650, 249, 630, 249),
        (480, 249, 400, 249),
        (325, 128, 325, 230),
        (250, 259, 200, 259),
    ]
    for x1, y1, x2, y2 in arrows:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>')
    defs = '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#333"/></marker></defs>'
    return _svg(width, height, defs + "\n" + "\n".join(parts))


def _short_mode(mode: str) -> str:
    return {
        "none": "none",
        "append_only_summary": "summary",
        "append_only_summary_with_rationale": "summary+rationale",
    }.get(mode, mode)


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
