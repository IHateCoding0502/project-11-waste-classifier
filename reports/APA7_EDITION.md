# APA 7 edition

The user's later request explicitly replaces the course's IEEE formatting for a
separate Word edition. `submission/*_report_APA7.docx` is the APA edition; the
original course report and review ZIP remain unchanged.

Run `scripts/build_apa_report.py` with the bundled document runtime, then update
fields and export an internal QA PDF with `scripts/render_submission_word.ps1`.
Run `scripts/verify_apa_report.py` after the Word refresh. Render every page with
Poppler and inspect page images before delivery.

APA formatting includes 12-point Times New Roman, double-spaced left-aligned
prose, half-inch first-line indents, one-inch margins, consecutive top-right page
numbers from the title page, unnumbered heading levels, author–date citations,
alphabetical hanging-indent references, bold table/figure numbers, italic titles,
horizontal table rules, and separate lettered appendices. Equations remain
editable Word math, with right-aligned numbers and corrected subscripts.

A4 paper is retained from the course specification. APA does not prescribe a table
of contents; the requested TOC uses compact spacing, heading indentation, dot
leaders, hyperlinks, and live page numbers. Code and table bodies use the allowed
more compact spacing. No contribution data, instructor name, exact due date, or
slide provenance has been invented. Existing supplied month/year is retained.

Style sources:
- https://apastyle.apa.org/instructional-aids/student-paper-setup-guide.pdf
- https://apastyle.apa.org/style-grammar-guidelines/paper-format/headings
- https://apastyle.apa.org/style-grammar-guidelines/references/elements-list-entry
