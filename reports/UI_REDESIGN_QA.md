# Streamlit UI redesign QA

Scope: presentation changes in starter/app.py, assets/streamlit.css and .streamlit/config.toml. No CNN architecture, inference implementation, preprocessing, trained checkpoint, guidance data or evaluation script was changed in this redesign.

- Automated checks: `python -m pytest tests -q --basetemp .qa_ui_final` — **39 passed in 17.31 seconds**. See ui_test_results.txt.
- The upload test additionally checks that changing the guidance category preserves a prediction and replacing the uploaded image clears its displayed scores before the next Predict action.
- Browser verification: all four sidebar pages open successfully; original model KPI values and architecture diagram render; limitation cards and high-confidence error gallery render; robustness cards display actual sample counts and separate pending real-camera/campus states.
- Actual browser upload: data/test/paper/paper100.jpg, JPEG 512×384. Preview appeared; Predict returned Paper with all six live model scores, confidence caveat, CUDA timing and the existing Paper guidance. The six progress indicators were inspected.
- Robustness examples expander opens correct/incorrect galleries, confusion matrix, per-class performance and confidence distribution.
- Responsive checks: desktop 1440×1000 and mobile 390×844. At 390 px, main cards stack, sidebar can expand/collapse, and document scrollWidth equals clientWidth (390); desktop also has no horizontal document overflow (1440). Mobile and desktop screenshots are saved beside this document. Temporary browser viewport override was reset after testing.
- No new dependencies or externally loaded fonts. Icons are local inline SVG; CSS uses app-owned classes, keyed containers and semantic Streamlit hooks rather than generated emotion class names.
- Selected checkpoint SHA-256 remains 825083ce60c04c0566b736a2089ece09f9a640ba0b72b55ce3b1fadf6ef077c3.

Application: http://127.0.0.1:8501
Screenshots: ui-desktop.jpg, ui-mobile.jpg.
