"""Saved evidence must stay correctly labeled and optional when artifacts are absent."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent.parent


def _page(name):
    return AppTest.from_string(f"from starter.dashboard_evidence import {name}\n{name}()").run(timeout=30)


def test_analytics_shows_saved_metrics_curves_and_pending_evaluation():
    selected = json.loads((ROOT / "results/selected.json").read_text(encoding="utf-8"))
    dashboard = _page("analytics_page")
    assert not dashboard.exception
    assert {metric.label: metric.value for metric in dashboard.metric} == {
        title: f"{selected['test'][field]:.1%}" for title, field in [
            ("Accuracy", "accuracy"), ("Macro precision", "macro_precision"),
            ("Macro recall", "macro_recall"), ("Macro F1", "macro_f1")]
    }
    assert len(dashboard.get("vega_lite_chart")) == 1
    dashboard.radio(key="analytics_view").set_value("Training").run(timeout=30)
    assert not dashboard.exception and len(dashboard.get("vega_lite_chart")) == 2
    dashboard.radio(key="analytics_view").set_value("Robustness").run(timeout=30)
    dashboard.selectbox(key="scenario").set_value("low_light_blur_real").run(timeout=30)
    assert not dashboard.exception and any("Awaiting genuine" in item.value for item in dashboard.info)


def test_guide_uses_material_record_and_observed_confusions():
    saved = json.loads((ROOT / "data/disposal_guidance.json").read_text(encoding="utf-8"))
    page = _page("guide_page")
    page.selectbox(key="guide_material").set_value("glass").run(timeout=30)
    assert not page.exception
    assert any(saved["glass"]["preparation"] in body.value for body in page.markdown)
    assert any("metal (16)" in block.value for block in page.info)


def test_missing_artifacts_are_reported(monkeypatch, tmp_path):
    from starter import dashboard_evidence
    monkeypatch.setattr(dashboard_evidence, "ROOT", tmp_path)
    page = _page("analytics_page")
    assert not page.exception and any("results/selected.json" in x.value for x in page.info)
    page = _page("guide_page")
    assert not page.exception and any("data/disposal_guidance.json" in x.value for x in page.info)
