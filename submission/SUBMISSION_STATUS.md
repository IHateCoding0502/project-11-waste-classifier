# AIN submission status

Prefix: **61FIT3AIN-TUT02_G05_P11**

Later formatting request: a separate `61FIT3AIN-TUT02_G05_P11_report_APA7.docx`
uses APA 7 author–date references and paper formatting. Its live TOC is one page.
The IEEE report and existing course review ZIP below are preserved; the APA
edition is not silently substituted into the IEEE-required submission package.
See `reports/APA7_EDITION.md` for intentional adaptations and verification.

Completed:

- English Word report authored from the supplied official template, preserving its A4 geometry, 2.5 cm margins, Times New Roman styles, university cover and native page fields.
- 22 total pages; exactly 15 main-body pages (Word pages 1-15; PDF verification pages 5-19).
- Updated live table of contents, ten native Word equations, eight live table captions, five live figure captions, IEEE references, Appendices A-C.
- Cover has all supplied member identities and the representative marker. Personal metadata and Word reports remain local and are excluded from Git.
- Python 3.11.16 environment installed successfully: 38 tests passed and one CUDA test skipped on the CPU build.
- Full 435-image standard re-evaluation under Python 3.11 reproduces 334 correct / accuracy 0.767816 / Macro-F1 0.757476. These compatibility results do not replace historical timings.
- Review ZIP created with exactly one matching root and slides/, code/, report/. The archive is below 50 MB; it contains no virtualenvs, Git metadata, full train/validation/test image data, secrets or extra report PDFs.
- The existing midterm PDF is copied byte-for-byte; it has not been regenerated or represented as a new final slide deck.
- A separate test run from the extracted code directory passed 38 tests, with one CUDA test skipped.

Still required before the ZIP can be called submission-ready:

1. Main tasks and agreed integer contribution percentages for all three members, totaling 100. Appendix B currently marks these entries as pending.
2. Confirmation that presentation.pdf is the exact previously presented midterm deck, or a path to the correct PDF. The current deck still contains an unfinished team-name line and historical pending-result notes; it is not silently edited because the guide requires the original presented version.

The default package command refuses an unready submission. `--review` deliberately produces an incomplete local review package with warnings in code/README.md. No file was submitted to an instructor or uploaded to a new service. The Word report and member metadata are local submission artifacts.
