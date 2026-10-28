#!/usr/bin/env python3
"""Build a seven-slide PDF focused on the current PyTorch Midterm AI Core.

Only artifacts marked ``framework=pytorch`` and ``data_mode=full_trashnet`` are
used for full-data performance claims. Without them the result slide explicitly
states that full-dataset metrics are pending; unsupported or historical scores
are never substituted for current PyTorch results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
W, H = 960, 540
NAVY = colors.HexColor("#12355B")
TEAL = colors.HexColor("#1B998B")
INK = colors.HexColor("#18212B")
MUTED = colors.HexColor("#5F6B7A")
PALE = colors.HexColor("#EEF3F8")
WHITE = colors.white
FONT, FONT_B = "Helvetica", "Helvetica-Bold"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
try:
    pdfmetrics.registerFont(TTFont("DVSans", str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DVSans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DVSans-Oblique", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFontFamily("DVSans", normal="DVSans", bold="DVSans-Bold", italic="DVSans-Oblique", boldItalic="DVSans-Bold")
    FONT, FONT_B = "DVSans", "DVSans-Bold"
except Exception:
    pass


def pstyle(size=14, color=INK, bold=False, leading=None):
    return ParagraphStyle("slide", fontName=FONT_B if bold else FONT, fontSize=size,
                          leading=leading or size * 1.28, textColor=color)


def paragraph(c, text, x, top, width, size=14, color=INK, bold=False, max_height=300):
    p = Paragraph(text, pstyle(size, color, bold))
    _, height = p.wrap(width, max_height)
    p.drawOn(c, x, top - height)
    return height


def bullet_list(c, items, x, top, width, size=15, gap=9, color=INK):
    y = top
    for item in items:
        h = paragraph(c, "• " + item, x, y, width, size, color)
        y -= h + gap
    return y


def draw_card(c, x, y, w, h, title, body=None):
    c.setFillColor(WHITE)
    c.setStrokeColor(colors.HexColor("#D5DEE8"))
    c.roundRect(x, y, w, h, 10, fill=1, stroke=1)
    paragraph(c, title, x + 16, y + h - 15, w - 32, 16, NAVY, True)
    if body:
        paragraph(c, body, x + 16, y + h - 42, w - 32, 12, MUTED)


def draw_header(c, title, slide_no, subtitle="Midterm milestone · AI Core"):
    c.setFillColor(PALE)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(NAVY)
    c.rect(0, H - 68, W, 68, stroke=0, fill=1)
    c.setFillColor(TEAL)
    c.rect(0, H - 72, W, 4, stroke=0, fill=1)
    c.setFillColor(WHITE)
    c.setFont(FONT_B, 25)
    c.drawString(34, H - 42, title)
    c.setFont(FONT, 10)
    c.setFillColor(colors.HexColor("#C9D8E8"))
    c.drawRightString(W - 34, H - 41, subtitle)
    c.setFillColor(MUTED)
    c.setFont(FONT, 9)
    c.drawString(34, 16, "Project 11 · Visual Waste Sorting Assistant")
    c.drawRightString(W - 34, 16, str(slide_no))


def add_image_fit(c, image_path: Path, x, y, w, h):
    if not image_path.is_file():
        return False
    img = ImageReader(str(image_path))
    iw, ih = img.getSize()
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(img, x + (w-dw)/2, y + (h-dh)/2, dw, dh, mask="auto")
    return True


def load_current_results(results_dir: Path):
    selected_path = results_dir / "selected.json"
    metadata_path = results_dir / "experiment_metadata.json"
    if not selected_path.is_file() or not metadata_path.is_file():
        return None
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("framework") != "pytorch" or metadata.get("data_mode") != "full_trashnet":
        return None
    if selected.get("framework") != "pytorch" or not isinstance(selected.get("test"), dict):
        return None
    return selected


def build(out: Path, team: str, results_dir: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    results = load_current_results(results_dir)
    c = canvas.Canvas(str(out), pagesize=(W, H))
    c.setTitle("Project 11 - Waste Classifier - PyTorch Midterm")
    c.setAuthor("Project 11 team")

    # 1. Cover
    c.setFillColor(NAVY); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(TEAL); c.rect(44, 110, 6, 300, fill=1, stroke=0)
    c.setFillColor(WHITE); c.setFont(FONT_B, 34)
    c.drawString(78, 360, "Visual Waste Sorting Assistant")
    paragraph(c, "A custom CNN implemented from scratch in PyTorch", 80, 318, 780, 21, colors.HexColor("#C9D8E8"))
    c.setFillColor(WHITE); c.setFont(FONT_B, 17); c.drawString(80, 244, "Midterm milestone — AI Core")
    c.setFillColor(colors.HexColor("#C9D8E8")); c.setFont(FONT, 14); c.drawString(80, 211, "TrashNet · Six-class image classification")
    c.setFont(FONT, 12); c.drawString(80, 165, team)
    c.setFont(FONT, 10); c.drawString(80, 42, "cardboard · glass · metal · paper · plastic · trash")
    c.showPage()

    # 2. Problem and data
    draw_header(c, "Task and dataset", 2)
    draw_card(c, 36, 298, 425, 132, "Objective", "Classify one waste-item image into one of six fixed TrashNet labels. The task is image classification, not object detection or a disposal-safety decision.")
    draw_card(c, 481, 298, 443, 132, "Input and output", "RGB image resized to 64 × 64. The model produces six logits in the canonical class order; Softmax is applied only for displayed probabilities.")
    paragraph(c, "Dataset protocol", 42, 266, 300, 19, NAVY, True)
    bullet_list(c, [
        "Pinned TrashNet archive; image validity checked before splitting.",
        "Class order is fixed: cardboard, glass, metal, paper, plastic, trash.",
        "Deterministic 70 / 13 / 17 train–validation–test split (seed 42).",
        "Hidden/AppleDouble metadata and invalid image files are excluded before split assignment.",
    ], 48, 232, 850, size=14, gap=6)
    c.showPage()

    # 3. Architecture and parameter count from live implementation
    draw_header(c, "CNN architecture", 3)
    from starter.student_core import build_model, count_parameters
    model = build_model((64, 64, 3), seed=42)
    params = count_parameters(model)
    labels = [
        ("Input", "64 × 64 × 3"), ("Conv block 1", "16 channels"), ("Conv block 2", "32 channels"),
        ("Conv block 3", "64 channels"), ("Conv block 4", "128 channels"), ("Global average pooling", "1 × 1 per channel"),
        ("Dropout → Dense 128", "ReLU + dropout"), ("Dense 6", "class logits"),
    ]
    x, y, bw, bh, gap = 66, 370, 320, 34, 10
    for i, (title, detail) in enumerate(labels):
        yy = y - i * (bh + gap)
        fill = NAVY if i in (0, len(labels)-1) else WHITE
        c.setFillColor(fill); c.setStrokeColor(TEAL if i == len(labels)-1 else colors.HexColor("#D5DEE8"))
        c.roundRect(x, yy, bw, bh, 6, fill=1, stroke=1)
        c.setFillColor(WHITE if fill == NAVY else NAVY); c.setFont(FONT_B, 11)
        c.drawString(x+12, yy+20, title)
        c.setFillColor(colors.HexColor("#DDE8F2") if fill == NAVY else MUTED); c.setFont(FONT, 9)
        c.drawRightString(x+bw-10, yy+9, detail)
        if i < len(labels)-1:
            c.setStrokeColor(TEAL); c.line(x+bw/2, yy-1, x+bw/2, yy-gap+1)
    draw_card(c, 438, 278, 483, 112, "Implementation", "Convolutional weights initialized from scratch. BatchNorm, ReLU and MaxPool are explicit modules. The model returns raw logits for CrossEntropyLoss; Softmax is not applied inside the training model.")
    draw_card(c, 438, 142, 483, 104, "Trainable parameter count", f"{params:,} parameters in the default architecture, computed directly from the current PyTorch model. No pretrained backbone or downloaded solution weights are used.")
    c.showPage()

    # 4. Training pipeline
    draw_header(c, "Training methodology", 4)
    bullet_list(c, [
        "Loss: class-weighted CrossEntropyLoss; weights are computed from training labels only.",
        "Optimizer: Adam with configurable initial learning rate.",
        "Scheduler: ReduceLROnPlateau monitors validation loss.",
        "Augmentation is applied only to training images; validation, test, and prediction use deterministic preprocessing.",
        "Best epoch is restored based on minimum validation loss.",
        "Python, NumPy and PyTorch RNGs are seeded; split manifest and configuration are recorded.",
    ], 52, 422, 820, size=16, gap=13)
    c.setFillColor(colors.HexColor("#D5F2EC")); c.roundRect(48, 63, 864, 66, 9, fill=1, stroke=0)
    paragraph(c, "Leakage control", 66, 112, 160, 15, NAVY, True)
    paragraph(c, "Model/configuration selection is validation-only. Test metrics are generated only for the final selected CNN and the required baselines.", 66, 88, 820, 13, INK)
    c.showPage()

    # 5. Experiment plan
    draw_header(c, "Baselines and controlled ablation", 5)
    draw_card(c, 42, 301, 408, 117, "Baseline 1", "Majority-class classifier: always predicts the most frequent training class.")
    draw_card(c, 472, 301, 444, 117, "Baseline 2", "Colour-histogram nearest-centroid classifier. Distance metric is selected on validation data.")
    paragraph(c, "One-factor-at-a-time CNN variations", 48, 270, 500, 18, NAVY, True)
    bullet_list(c, [
        "Augmentation: none / geometric control / geometric + photometric.",
        "Dropout: 0.0 / 0.3 control / 0.5; keep `no_reg` as an explicit ablation.",
        "Depth: 2 / 3 / 4 convolutional blocks.",
        "Learning rate: 2e-4 / 2e-3 control / 1e-2.",
    ], 54, 238, 840, size=14, gap=5)
    c.setFillColor(PALE); c.roundRect(48, 65, 860, 48, 6, fill=1, stroke=0)
    paragraph(c, "The final demo model is selected among configurations with dropout > 0; no_reg is an ablation only. Compare one changed factor at a time and use learning curves to explain underfitting or overfitting.", 62, 98, 830, 13, INK)
    c.showPage()

    # 6. Results with verified data only
    draw_header(c, "Results and model selection", 6)
    if results:
        selected_name = results["selected"]
        test = results["test"]
        ci = results.get("test_ci95") or {}
        lo, hi = (ci.get("macro_f1") or [None, None])
        data = [
            ["Model", "Test accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"],
            ["Selected PyTorch CNN: " + selected_name, f"{test['accuracy']:.3f}", f"{test['macro_precision']:.3f}", f"{test['macro_recall']:.3f}", f"{test['macro_f1']:.3f}"],
        ]
        t = Table(data, colWidths=[300, 115, 130, 120, 110], rowHeights=[32, 42])
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
                               ("FONTNAME", (0,0), (-1,0), FONT_B), ("FONTNAME", (0,1), (-1,-1), FONT),
                               ("FONTSIZE", (0,0), (-1,-1), 10), ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CDD7E2")),
                               ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("BACKGROUND", (0,1), (-1,1), WHITE)]))
        t.wrapOn(c, 800, 100); t.drawOn(c, 48, 318)
        paragraph(c, f"Selected by validation macro-F1; best epoch = {results.get('best_epoch', 'see selected.json')}.", 52, 295, 850, 14, NAVY, True)
        if lo is not None:
            paragraph(c, f"Test macro-F1 bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]. This interval reflects resampling of test images, not variability across random seeds.", 52, 263, 850, 12, MUTED)
        cm_path = results_dir / "figures" / "confusion_matrix.png"
        summary_path = results_dir / "figures" / "summary_macro_f1.png"
        add_image_fit(c, summary_path, 60, 58, 420, 190)
        add_image_fit(c, cm_path, 510, 50, 365, 205)
    else:
        draw_card(c, 52, 304, 856, 103, "Full TrashNet results: pending", "The offline source package contains only synthetic smoke-test images. Container network access prevented download of the pinned TrashNet archive, so this run does not claim full-dataset performance metrics.")
        bullet_list(c, [
            "The PyTorch implementation is tested using the offline sample pipeline.",
            "Run dataset preparation and the full experiment suite to populate this slide with actual PyTorch metrics.",
            "Historical TensorFlow/Keras numbers are intentionally not used as PyTorch results.",
        ], 66, 278, 825, size=14, gap=10)
        c.setFillColor(colors.HexColor("#FFF1D8")); c.roundRect(52, 65, 856, 55, 8, fill=1, stroke=0)
        paragraph(c, "Do not quote this status slide as final model evaluation. Full-data metrics must be generated on the pinned TrashNet split.", 68, 101, 820, 14, colors.HexColor("#7A4B00"), True)
    c.showPage()

    # 7. Conclusions and limitations
    draw_header(c, "Conclusion and limitations", 7)
    bullet_list(c, [
        "The pipeline is designed around a compact, from-scratch CNN with class-weighted loss and validation-based checkpointing.",
        "The majority and colour-histogram baselines establish simple reference points; controlled ablations examine augmentation, dropout, depth and learning rate.",
        "The no_reg run is an ablation, not proof that regularization is unnecessary. Interpret it alongside training–validation gaps.",
        "Small test sets and a single seed limit confidence in small differences between configurations.",
        "The model covers only six TrashNet classes; confidence is not a calibrated guarantee and an image can be out of distribution.",
    ], 52, 424, 840, size=15, gap=10)
    c.setFillColor(NAVY); c.roundRect(52, 65, 856, 60, 8, fill=1, stroke=0)
    paragraph(c, "Next step before submission: prepare the full dataset, run the complete PyTorch experiments, rebuild this deck, and verify the live demo with the selected checkpoint.", 68, 105, 820, 14, WHITE, True)
    c.save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", default="Team name and member names: add before submission")
    parser.add_argument("--out", type=Path, default=ROOT / "presentation.pdf")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    build(args.out, args.team, args.results_dir)
    print(f"Wrote {args.out}. Data status: {'full PyTorch results' if load_current_results(args.results_dir) else 'full-dataset metrics pending'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
