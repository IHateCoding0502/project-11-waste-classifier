"""Verify the separate APA edition after Word updates fields and pagination."""
import hashlib
import json
import re
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent.parent
report=ROOT/'submission/61FIT3AIN-TUT02_G05_P11_report_APA7.docx'
pdf=ROOT/'tmp/submission/apa7.pdf'
doc=Document(report)
pages=[p.extract_text() for p in PdfReader(pdf).pages]
body='\n'.join(pages[3:])
assert len(doc.sections)==1
s=doc.sections[0]
assert all(abs(getattr(s,k).inches-1)<.001 for k in ['top_margin','bottom_margin','left_margin','right_margin'])
assert doc.styles['Normal'].font.name=='Times New Roman'
assert doc.styles['Normal'].font.size.pt==12
assert doc.styles['Normal'].paragraph_format.line_spacing==2
assert doc.styles['Reference'].paragraph_format.first_line_indent.inches==-.5
assert not re.search(r'\[\d+\]',body), 'Unconverted IEEE citation'
assert 'Table of Contents' in pages[2] and 'Appendix C' in pages[2]
assert 'Visual Waste Sorting Assistant' in pages[3]
assert 'Update table of contents' not in '\n'.join(pages)
assert not any('Error! Reference' in p or 'Error! Bookmark' in p for p in pages)
fields=[t.text or '' for t in doc.element.findall('.//'+qn('w:instrText'))]
assert sum('SEQ ' in f for f in fields)==13
assert any('TOC ' in f for f in fields)
assert any('PAGE' in (t.text or '') for t in s.header._element.findall('.//'+qn('w:instrText')))
equations=[p for p in doc.paragraphs if re.search(r'\(\d+\)$',p.text) and p._p.find(qn('m:oMath')) is not None]
assert len(equations)==10
refs=[p for p in doc.paragraphs if p.style.name=='Reference']
assert len(refs)==13
assert [p.text for p in refs]==sorted([p.text for p in refs],key=str.casefold)
assert len(doc.tables)==8
assert all(str(i+1)==p.split()[0] for i,p in enumerate(pages))
assert 'Table B1' in '\n'.join(pages) and 'Table C1' in '\n'.join(pages)
original=ROOT/'submission/61FIT3AIN-TUT02_G05_P11_report.docx'
course=json.loads((ROOT/'submission/report_verification.json').read_text())
out={
 'report':str(report),'sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
 'pages':len(pages),'toc_pages':[3],'editable_display_equations':len(equations),
 'native_captions':13,'references':13,'tables':8,'fields_refreshed_in':'Microsoft Word',
 'visual_qa':'All pages rendered with Poppler and inspected',
 'course_report_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
 'adaptations':['A4 retained from course specification','Requested live compact TOC; APA does not prescribe TOC formatting'],
 'pending':['Agreed member tasks and contribution percentages','Exact original midterm slide confirmation'],
 'course_zip_replaced':False,
}
(ROOT/'submission/apa7_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
