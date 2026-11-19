"""Create an APA 7 edition without overwriting the course's IEEE edition.

Use the bundled document runtime. The technical content and editable OMML are
shared with the course report; APA styles and author-date references are separate.
"""
from pathlib import Path
from copy import deepcopy
import json
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml.ns import qn
from build_submission_report import Builder, ROOT, FIG, element, equation_nodes, sub, mr, fraction, summation
from submission_report_content import write_report

TITLE = 'Visual Waste Sorting Assistant'
CITES = {
    1:'Thung & Yang, 2016', 2:'Paszke et al., 2019', 3:'PyTorch contributors, n.d.',
    4:'Ioffe & Szegedy, 2015', 5:'Srivastava et al., 2014', 6:'Guo et al., 2017',
    7:'Streamlit, n.d.', 8:'Pillow contributors, n.d.', 9:'Harris et al., 2020',
    10:'Hunter, 2007', 11:'U.S. Environmental Protection Agency, n.d.',
    12:'Vietnam Environment Agency, 2024', 13:'Kingma & Ba, 2015',
}
# Asterisks identify italic spans, not Markdown in the delivered document.
REFERENCES = [
 'Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In D. Precup & Y. W. Teh (Eds.), *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html',
 'Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., Fernández del Río, J., Wiebe, M., Peterson, P., . . . Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2',
 'Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55',
 'Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In F. Bach & D. Blei (Eds.), *Proceedings of the 32nd International Conference on Machine Learning* (Vol. 37, pp. 448–456). PMLR. https://proceedings.mlr.press/v37/ioffe15.html',
 'Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization* [Conference paper]. International Conference on Learning Representations. https://arxiv.org/abs/1412.6980',
 'Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., . . . Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. Fox, & R. Garnett (Eds.), *Advances in neural information processing systems* (Vol. 32). Curran Associates. https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library',
 'Pillow contributors. (n.d.). *Pillow 13.0.0 documentation*. Retrieved November 19, 2026, from https://pillow.readthedocs.io/en/stable/',
 'PyTorch contributors. (n.d.). *Conv2d*. PyTorch documentation. Retrieved November 19, 2026, from https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html',
 'Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html',
 'Streamlit. (n.d.). *Streamlit documentation*. Retrieved November 19, 2026, from https://docs.streamlit.io/',
 'Thung, G., & Yang, M. (2016). *TrashNet* [Data set]. GitHub. https://github.com/garythung/trashnet',
 'U.S. Environmental Protection Agency. (n.d.). *How do I recycle common recyclables*. Retrieved November 19, 2026, from https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables',
 'Vietnam Environment Agency. (2024, April 5). *Phân loại rác thải tại nguồn* [Source separation of waste]. https://vea.mae.gov.vn/tin-tuc-su-kien/10526/phan-loai-rac-thai-tai-nguo',
]

def citation_text(text):
    # Collapse adjacent numerical references into one alphabetized APA citation.
    def replace(match):
        numbers = [int(n) for n in re.findall(r'\[(\d+)\]', match.group())]
        return '(' + '; '.join(sorted({CITES[n] for n in numbers})) + ')'
    text = re.sub(r'\[\d+\](?:,?\s*\[\d+\])*', replace, text)
    return text.replace('Section 2.3 discussion', 'the Baseline and Related Work discussion').replace('64 x 64', '64 × 64')

class APABuilder(Builder):
    def __init__(self, template, team):
        super().__init__(template, team)
        self.doc = Document()
        self.ref_written = False
        self.appendix = None
        self.appendix_tables = 0
        self.section = self.doc.sections[0]
        # A4 is retained as the explicit course requirement; APA permits adaptations.
        self.section.page_width = Inches(8.2677)
        self.section.page_height = Inches(11.6929)
        self.section.top_margin = self.section.bottom_margin = Inches(1)
        self.section.left_margin = self.section.right_margin = Inches(1)
        self.section.header_distance = self.section.footer_distance = Inches(.5)
        styles = self.doc.styles
        for name in ['Normal','Title','Heading 1','Heading 2','Heading 3','Caption','TOC Heading']:
            s = styles[name]
            s.font.name = 'Times New Roman'; s.font.size = Pt(12)
            s.font.color.rgb = RGBColor(0,0,0)
            f = s.paragraph_format
            f.line_spacing = 2; f.space_before = f.space_after = Pt(0)
            f.first_line_indent = Inches(0)
            f.widow_control = True
        f = styles['Normal'].paragraph_format
        f.first_line_indent = Inches(.5); f.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for name in ['Title','Heading 1','Heading 2','Heading 3','TOC Heading']:
            styles[name].font.bold = True
            styles[name].paragraph_format.keep_with_next = True
        for name in ['Title','Heading 1','TOC Heading']:
            styles[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        styles['Heading 3'].font.italic = True
        for name in ['Reference','Code','Front Heading','TOC 1','TOC 2']:
            if name not in styles: styles.add_style(name, 1)
            styles[name].base_style = styles['Normal']
            styles[name].font.name = 'Times New Roman'; styles[name].font.size = Pt(12)
            styles[name].paragraph_format.first_line_indent = Inches(0)
        styles['Front Heading'].base_style = styles['Heading 1']
        styles['Reference'].paragraph_format.left_indent = Inches(.5)
        styles['Reference'].paragraph_format.first_line_indent = Inches(-.5)
        styles['Code'].font.name = 'Consolas'; styles['Code'].font.size = Pt(9)
        styles['Code'].paragraph_format.line_spacing = 1
        for i in [1,2]:
            styles[f'TOC {i}']._element.attrib.pop(qn('w:customStyle'),None)
            f = styles[f'TOC {i}'].paragraph_format
            f.line_spacing = 1; f.space_after = Pt(2)
            f.left_indent = Inches(.22 if i==2 else 0)
            f.first_line_indent = Inches(0)
            styles[f'TOC {i}'].font.bold = False
            styles[f'TOC {i}']._element.find(qn('w:name')).set(qn('w:val'),f'toc {i}')
        styles['Caption'].font.bold=False
        styles['Caption'].font.italic=False
        # Strip theme overrides and the stock Title border from python-docx's base.
        for s in styles:
            for fonts in s._element.findall('.//'+qn('w:rFonts')):
                for key in list(fonts.attrib):
                    if 'theme' in key.lower():del fonts.attrib[key]
            for border in s._element.findall('.//'+qn('w:pBdr')):border.getparent().remove(border)
        for name in ['Front Heading','TOC Heading']:
            pr=styles[name]._element.get_or_add_pPr()
            pr.append(element('w:outlineLvl',**{'w:val':'9'}))
        p = self.section.header.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.paragraph_format.first_line_indent = Inches(0)
        p.add_run()._r.append(element('w:fldChar', **{'w:fldCharType':'begin'}))
        p.add_run()._r.append(element('w:instrText',' PAGE '))
        p.add_run()._r.append(element('w:fldChar', **{'w:fldCharType':'end'}))
        # OMML uses Cambria Math; prose remains Times New Roman.
        mp = self.doc.settings.element.find(qn('m:mathPr'))
        if mp is None: mp=element('m:mathPr'); self.doc.settings.element.append(mp)
        mp.append(element('m:mathFont', **{'m:val':'Cambria Math'}))

    def add_node(self, node):
        self.doc.element.body.sectPr.addprevious(node)
        return node

    def page(self):
        # Remove the old 15-page hard pagination; let APA text flow naturally.
        pass

    def p(self, text, style='Normal', bold=False):
        if style == 'Reference':
            if not self.ref_written:
                self.ref_written=True
                for ref in REFERENCES:
                    p=self.doc.add_paragraph(style='Reference')
                    for i,part in enumerate(ref.split('*')):p.add_run(part).italic=bool(i%2)
            return None
        p=self.doc.add_paragraph(style=style)
        text=citation_text(text)
        # Replace pseudo-subscripts in prose with editable inline mathematics.
        tokens={'(xᵢ,yᵢ)':[mr('('),sub('x','i'),mr(', '),sub('y','i'),mr(')')],
                'xᵢ':[sub('x','i')], 'yᵢ':[sub('y','i')], 'fθ':[sub('f','θ')],
                'nₖ':[sub('n','k')], 'Ntrain/(6nₖ)':[mr('N')],
                'sᵢ,yᵢ':[sub('s','i,yᵢ')], 'Mₐ,b':[sub('M','a,b')]}
        # Render the complete class-weight expression rather than partial glyphs.
        from build_submission_report import fraction
        tokens['Ntrain/(6nₖ)']=[fraction([sub('N','train')],[mr('6'),sub('n','k')])]
        pattern='('+'|'.join(re.escape(t) for t in sorted(tokens,key=len,reverse=True))+')'
        for part in re.split(pattern,text):
            if part in tokens:
                math=element('m:oMath')
                for node in tokens[part]:math.append(deepcopy(node))
                p._p.append(math)
            else:p.add_run(part).bold=bold if bold else None
        return p

    def h(self, text, level=2):
        text=re.sub(r'^\d+(?:\.\d+)*\s+', '', text)
        text=text.replace('Limitations Ethics and Conclusion','Limitations Ethics and Conclusion')
        if text.startswith('Introduction and Problem Definition'):text=TITLE
        if text=='References' or text.startswith('Appendix '):
            self.doc.add_page_break()
        if text.startswith('Appendix '):
            self.appendix=text[9]; self.appendix_tables=0
            label, title = text[:10], text[10:].strip()
            p=self.doc.add_paragraph(style='Heading 1')
            p.add_run(label).add_break();p.add_run(title)
            return p
        return self.p(text,f'Heading {level}')

    def front(self, abstract):
        for _ in range(3):self.doc.add_paragraph()
        self.p(TITLE,'Title')
        self.doc.add_paragraph()
        names=[m['name'] for m in self.team['members']]
        authors=', '.join(names[:-1])+', and '+names[-1]
        for text in [authors,
                     'Hanoi University', self.team['class_code'],
                     'Group 05 · Project 11', 'November 2026']:
            p=self.p(text);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent=Inches(0)
        self.doc.add_page_break()
        self.p('Abstract','Front Heading')
        p=self.p(abstract);p.paragraph_format.first_line_indent=Inches(0)
        p=self.doc.add_paragraph();p.add_run('Keywords: ').italic=True
        p.add_run('waste classification, convolutional neural networks, robustness, Streamlit, reproducibility')
        self.doc.add_page_break();self.p('Table of Contents','TOC Heading')
        p=self.doc.add_paragraph()
        for typ in ['begin']:
            p.add_run()._r.append(element('w:fldChar',**{'w:fldCharType':typ}))
        p.add_run()._r.append(element('w:instrText',' TOC \\o "1-2" \\h \\z \\u '))
        p.add_run()._r.append(element('w:fldChar',**{'w:fldCharType':'separate'}))
        p.add_run('Update table of contents')
        p.add_run()._r.append(element('w:fldChar',**{'w:fldCharType':'end'}))
        self.doc.add_page_break()

    def caption(self, kind, text):
        if kind=='Table':
            if self.appendix:
                self.appendix_tables+=1;n=f'{self.appendix}{self.appendix_tables}'
                sequence=f'Table{self.appendix}'
            else:self.table_no+=1;n=str(self.table_no);sequence='Table'
        else:self.figure_no+=1;n=str(self.figure_no);sequence='Figure'
        title, _, note=text.partition('. ')
        title=title.rstrip('.')
        # APA title case, with common short function words left lowercase.
        small={'a','an','and','as','at','by','for','in','of','on','or','the','to','with'}
        title=' '.join(w if i and w.lower() in small else w[0].upper()+w[1:] for i,w in enumerate(title.split()))
        p=self.doc.add_paragraph(style='Caption');p.paragraph_format.keep_with_next=True
        p.add_run(kind+' '+(self.appendix if kind=='Table' and self.appendix else '')).bold=True
        for typ in ['begin']:
            p.add_run()._r.append(element('w:fldChar',**{'w:fldCharType':typ}))
        p.add_run()._r.append(element('w:instrText',f' SEQ {sequence} \\* ARABIC '))
        p.add_run()._r.append(element('w:fldChar',**{'w:fldCharType':'separate'}))
        p.add_run(re.sub(r'^[BC]','',n)).bold=True
        p.add_run()._r.append(element('w:fldChar',**{'w:fldCharType':'end'}))
        p=self.doc.add_paragraph(style='Caption');p.paragraph_format.keep_with_next=True
        p.add_run(title).italic=True
        return note

    def note(self,text):
        if not text:return
        p=self.doc.add_paragraph(style='Caption');p.add_run('Note. ').italic=True;p.add_run(citation_text(text))

    def table(self, headers, rows, caption, sample=4, best_rows=()):
        note=self.caption('Table',caption)
        table=self.doc.add_table(rows=1, cols=len(headers))
        table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
        n=len(headers)
        ratios={2:[.23,.77],3:[.34,.33,.33],4:[.37,.14,.24,.25]}.get(n,[1/n]*n)
        if n==4 and headers[0]=='Student ID':ratios=[.20,.27,.42,.11]
        if n==3 and headers[0]=='Method':ratios=[.52,.24,.24]
        if n==3 and headers[0]=='Tool':ratios=[.22,.40,.38]
        for col,ratio in zip(table.columns,ratios):col.width=Inches(6.2677*ratio)
        for i, values in enumerate([headers]+list(rows)):
            row=table.rows[0] if i==0 else table.add_row()
            trpr=row._tr.get_or_add_trPr();trpr.append(element('w:cantSplit'))
            if i==0:trpr.append(element('w:tblHeader'))
            for j,(cell,value) in enumerate(zip(row.cells,values)):
                cell.width=Inches(6.2677*ratios[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                p=cell.paragraphs[0];p.paragraph_format.first_line_indent=Inches(0)
                p.paragraph_format.line_spacing=1.15;p.paragraph_format.space_after=Pt(5);p.paragraph_format.space_before=Pt(5)
                p.paragraph_format.keep_with_next=False
                if len(rows)<=5 and i<len(rows):p.paragraph_format.keep_with_next=True
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT
                run=p.add_run(citation_text(str(value)));run.font.size=Pt(11)
                run.bold=(i==0 or (i-1 in best_rows and j>0))
                borders=element('w:tcBorders')
                for side in ['top','bottom','left','right','insideH','insideV']:
                    visible=(side=='top' and i==0) or (side=='bottom' and (i==0 or i==len(rows)))
                    borders.append(element('w:'+side,**{'w:val':'single' if visible else 'nil','w:sz':'6','w:color':'000000'}))
                cell._tc.get_or_add_tcPr().append(borders)
        self.note(note)

    def image(self,name,width,caption):
        note=self.caption('Figure',caption)
        p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent=Inches(0)
        p.paragraph_format.line_spacing=1
        p.paragraph_format.keep_with_next=bool(note)
        if name=='mobile.png':width=1.9
        p.add_run().add_picture(str(FIG/name),width=Inches(min(width,6.2677)))
        self.note(note)

    def equation(self,name):
        self.eq+=1
        p=self.doc.add_paragraph();p.paragraph_format.first_line_indent=Inches(0)
        if name in {'accuracy','pr'}:p.paragraph_format.keep_with_next=True
        p.paragraph_format.tab_stops.add_tab_stop(Inches(3.1338),WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(6.2677),WD_TAB_ALIGNMENT.RIGHT)
        p.add_run('\t');math=element('m:oMath')
        nodes=equation_nodes(name)
        if name=='f1':
            nodes=[sub('F','1,k'),mr(' = '),fraction([mr('2'),sub('P','k'),sub('R','k')],[sub('P','k'),mr(' + '),sub('R','k')]),mr(', '),sub('F','1,macro'),mr(' = '),fraction([mr('1')],[mr('6')]),summation('k=0,…,5',[sub('F','1,k')])]
        for node in nodes:math.append(node)
        # Operators and metric abbreviations are upright; variables are italic.
        for r in math.findall('.//'+qn('m:r')):
            text=''.join(r.itertext())
            if any(s in text for s in ['exp','log','max','Bernoulli','Accuracy','TP','FP','FN','F1','out','in','macro','train']):
                pr=element('m:rPr');pr.append(element('m:sty',**{'m:val':'p'}));r.insert(0,pr)
        p._p.append(math);p.add_run(f'\t({self.eq})')

    def save(self,out):
        paragraphs=self.doc.paragraphs
        for i,p in enumerate(paragraphs[:-1]):
            if p.style.name=='Code':
                p.paragraph_format.keep_with_next=paragraphs[i+1].style.name=='Code'
        # Give each table/figure a double-spaced separation from following prose.
        children=list(self.doc.element.body)
        for i,node in enumerate(children[:-1]):
            if node.tag==qn('w:tbl'):
                following=children[i+1]
                if following.tag==qn('w:p'):
                    from docx.text.paragraph import Paragraph
                    Paragraph(following,self.doc).paragraph_format.space_before=Pt(12)
        # Compact only the reproducibility appendix's explanatory ledger; retain
        # double spacing in all prose and avoid an almost empty overflow page.
        for p in self.doc.paragraphs:
            if p.text.startswith('Repository: https://github.com/'):
                p.text=('Repository: https://github.com/IHateCoding0502/project-11-waste-classifier. '
                        'Use Python 3.11 and the retained starter layout. Install dependencies before '
                        'offline operation; inference uses the locally trained CNN checkpoint.')
            elif p.text.startswith('For tests, install requirements-dev.txt'):
                p.text='Install the development requirements to run the checks below. Synthetic test fixtures verify software behavior only.'
            elif p.text.startswith('For full-data reproduction, obtain'):
                p.text=('For training and full-data evaluation, obtain the pinned TrashNet archive as '
                        'documented in data/LICENSE.md. The ZIP excludes the full training, validation, '
                        'and test folders. Run these commands after preparing the source images.')
            if p.text.startswith('Tables 1–2 document'):
                p.text=('Result sources: Tables 3–4 use results/baselines.json, selected.json, and run_*.json; '
                        'Tables 5–6 use results/stress/metrics.json. Figures 1–2 show the application, Figure 3 '
                        'shows its architecture, and Figures 4–5 use saved stress results and source images. '
                        'Numeric evidence remains available in machine-readable files.')
            if p.text.startswith('Evaluations record checkpoint, manifest'):
                p.text=('Evaluation records include checkpoint, manifest, and image hashes. Hardware changes '
                        'may prevent bit-for-bit retraining reproducibility; retain newly measured results.')
        self.doc.settings.element.append(element('w:updateFields',**{'w:val':'true'}))
        self.doc.core_properties.title=TITLE
        self.doc.core_properties.subject='APA 7 edition of the waste classification project report'
        self.doc.core_properties.author='; '.join(m['name'] for m in self.team['members'])
        out.parent.mkdir(parents=True,exist_ok=True);self.doc.save(out)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--team',type=Path,default=ROOT/'submission/team.json')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    team=json.loads(args.team.read_text(encoding='utf-8-sig'))
    builder=APABuilder(args.template,team)
    write_report(builder)
    out=args.out or ROOT/'submission'/f'{builder.prefix}_report_APA7.docx'
    builder.save(out)
    print(out)
