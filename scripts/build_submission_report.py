"""Build the English AIN report by editing a copy of the official DOCX template.
Run with a Python environment containing python-docx and Pillow.
Original template package parts are preserved unless explicitly edited.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
from lxml import etree
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent.parent
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
FIG=ROOT/'reports/submission_figures'


def element(tag,text=None,**attrs):
    e=OxmlElement(tag)
    if text is not None:
        e.text=text
        if text != text.strip():e.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    for k,v in attrs.items():e.set(qn(k),str(v))
    return e

def mr(text):
    r=element('m:r');r.append(element('m:t',text));return r

def sub(base,index):
    e=element('m:sSub');b=element('m:e');b.append(mr(base));s=element('m:sub');s.append(mr(index));e.extend([b,s]);return e

def sup(base,power):
    e=element('m:sSup');b=element('m:e');b.append(mr(base));s=element('m:sup');s.append(mr(power));e.extend([b,s]);return e

def subsup(base,index,power):
    e=element('m:sSubSup');b=element('m:e');b.append(mr(base))
    lo=element('m:sub');lo.append(mr(index));hi=element('m:sup');hi.append(mr(power))
    e.extend([b,lo,hi]);return e

def limit(operator,index):
    e=element('m:limLow');base=element('m:e');base.append(mr(operator))
    lo=element('m:lim');lo.append(mr(index));e.extend([base,lo]);return e

def fraction(num,den):
    e=element('m:f');a=element('m:num');b=element('m:den')
    for node in num:a.append(node)
    for node in den:b.append(node)
    e.extend([a,b]);return e

def summation(index,expression):
    n=element('m:nary');pr=element('m:naryPr');pr.append(element('m:chr',**{'m:val':'∑'}));pr.append(element('m:limLoc',**{'m:val':'subSup'}));pr.append(element('m:supHide',**{'m:val':'1'}));n.append(pr)
    lo=element('m:sub');lo.append(mr(index));n.append(lo);n.append(element('m:sup'));body=element('m:e')
    for v in expression:body.append(v)
    n.append(body);return n

def root(expr):
    r=element('m:rad');pr=element('m:radPr');pr.append(element('m:degHide',**{'m:val':'1'}));r.append(pr);r.append(element('m:deg'));b=element('m:e')
    for node in expr:b.append(node)
    r.append(b);return r

def equation_nodes(name):
    if name=='formulation':return [sub('z','i'),mr(' = '),sub('f','θ'),mr('('),sub('x','i'),mr(') ∈ '),sup('ℝ','6'),mr(',  ŷ = '),limit('arg max','k'),mr(' '),sub('z','k')]
    if name=='softmax':return [sub('s','k'),mr(' = '),fraction([mr('exp('),sub('z','k'),mr(' − a)')],[summation('j=0,…,5',[mr('exp('),sub('z','j'),mr(' − a)')])]),mr(', a = '),limit('max','j'),mr(' '),sub('z','j')]
    if name=='loss':return [sub('ℒ','B'),mr(' = −'),fraction([summation('i∈B',[sub('w','yᵢ'),mr(' log '),sub('s','i,yᵢ')])],[summation('i∈B',[sub('w','yᵢ')])])]
    if name=='conv':return [sub('Z','o,i,j'),mr(' = '),summation('c,u,v',[sub('W','o,c,u,v'),mr(' '),sub('X','c,si+u−p,sj+v−p')])]
    if name=='dimensions':return [sub('H','out'),mr(' = ⌊'),fraction([sub('H','in'),mr(' + 2p − K')],[mr('s')]),mr('⌋ + 1')]
    if name=='bn':return [mr('a = γ '),fraction([mr('z − '),sub('μ','B')],[root([subsup('σ','B','2'),mr(' + ε')])]),mr(' + β')]
    if name=='dropout':return [sub('h̃','j'),mr(' = '),fraction([sub('m','j'),mr(' '),sub('h','j')],[mr('1 − p')]),mr(', '),sub('m','j'),mr(' ∼ Bernoulli(1 − p)')]
    if name=='accuracy':return [mr('Accuracy = '),fraction([summation('k',[sub('M','k,k')])],[summation('a,b',[sub('M','a,b')])])]
    if name=='pr':return [sub('P','k'),mr(' = '),fraction([sub('TP','k')],[sub('TP','k'),mr(' + '),sub('FP','k')]),mr(', '),sub('R','k'),mr(' = '),fraction([sub('TP','k')],[sub('TP','k'),mr(' + '),sub('FN','k')])]
    if name=='f1':return [sub('F1','k'),mr(' = '),fraction([mr('2'),sub('P','k'),sub('R','k')],[sub('P','k'),mr(' + '),sub('R','k')]),mr(', '),sub('F1','macro'),mr(' = '),fraction([mr('1')],[mr('6')]),summation('k=0,…,5',[sub('F1','k')])]
    raise ValueError(name)


def make_figures():
    FIG.mkdir(parents=True,exist_ok=True)
    # Grayscale copies are derived only for the required black-and-white report.
    for source,name in [(ROOT/'reports/ui-desktop.jpg','studio'),(ROOT/'reports/ui-mobile.jpg','mobile'),(ROOT/'results/stress/standard_confusion.png','confusion')]:
        with Image.open(source) as image:
            gray=image.convert('L').convert('RGB')
        draw=ImageDraw.Draw(gray)
        try:font=ImageFont.truetype('arialbd.ttf',24)
        except OSError:font=ImageFont.load_default()
        points={'studio':[(30,103,'A'),(358,292,'B'),(356,542,'C'),(900,542,'D')],'mobile':[(9,90,'A'),(10,365,'B')]}.get(name,[])
        for x,y,letter in points:
            draw.rectangle((x,y,x+33,y+33),fill='white',outline='black',width=2);draw.text((x+7,y+2),letter,fill='black',font=font)
        gray.save(FIG/f'{name}.png')
    def font(size=24):
        try:return ImageFont.truetype('arial.ttf',size)
        except OSError:return ImageFont.load_default()
    image=Image.new('RGB',(1500,260),'white');draw=ImageDraw.Draw(image)
    labels=['Streamlit UI\nupload + preview','Validated image\nEXIF / RGB','Original CNN\n64 x 64 / 255','Six scores\n+ limitations','Guidance data\n+ source links']
    for i,label in enumerate(labels):
        x=10+i*300;draw.rounded_rectangle((x,55,x+265,190),radius=12,outline='black',width=3)
        draw.multiline_text((x+132,122),label,fill='black',font=font(24),anchor='mm',align='center',spacing=8)
        if i<4:
            draw.line((x+268,122,x+295,122),fill='black',width=3);draw.polygon([(x+295,122),(x+285,115),(x+285,129)],fill='black')
    image.save(FIG/'system.png')
    preds=json.loads((ROOT/'results/stress/predictions.json').read_text())
    errors=sorted([r for r in preds if r['condition']=='standard' and r['prediction']!=r['true_label']],key=lambda r:r['confidence'],reverse=True)[:2]
    image=Image.new('RGB',(1100,480),'white');draw=ImageDraw.Draw(image)
    for i,r in enumerate(errors):
        with Image.open(ROOT/r['path']) as im:tile=im.convert('L').convert('RGB')
        tile.thumbnail((500,360));image.paste(tile,(i*550+(550-tile.width)//2,100))
        text=f"{Path(r['path']).name}\n{r['true_label']} -> {r['prediction']}; score {r['confidence']:.2%}"
        draw.multiline_text((i*550+275,43),text,fill='black',font=font(22),anchor='mm',align='center',spacing=6)
    image.save(FIG/'errors.png')



class Builder:
    def __init__(self,template,team):
        self.template=template;self.team=team;self.doc=Document(template)
        self.prefix=f"{team['class_code']}_G{team['group']:02d}_P{team['project']:02d}"
        self.eq=0;self.table_no=0;self.figure_no=0
        self.original_children=list(self.doc.element.body)
        self.table_samples=[deepcopy(t._tbl) for t in self.doc.tables]
        self.caption_sample=deepcopy(self.doc.paragraphs[35]._p)
        self.equation_sample=deepcopy(self.doc.paragraphs[47]._p)
        # Edit title and member slots in place, preserving the cover logo and border.
        for p in self.doc.paragraphs:
            if p.text=='[PROJECT TITLE]':self.replace_text(p._p,'Visual Waste Sorting Assistant')
            elif p.text.startswith('Project [P00]'):self.replace_text(p._p,f"Project P11  ·  Group G{team['group']:02d}  ·  Class {team['class_code']}")
            elif p.text.startswith('Hanoi, [Month'):self.replace_text(p._p,'Hanoi, November 2026')
        members=team['members']
        self.fill_table(self.doc.tables[0]._tbl,['No.','Full name','Student ID'],[[i+1,m['name']+('*' if m['representative'] else ''),m['student_id']] for i,m in enumerate(members)])
        # Retain cover and its first section break only; reconstruct approved slots.
        for child in list(self.doc.element.body)[12:]:self.doc.element.body.remove(child)
        self.front_break=deepcopy(self.original_children[31])
        self.final_sect=deepcopy(self.original_children[-1])
        self.doc.element.body.append(self.final_sect)
        self.toc=deepcopy(self.original_children[29])

    @staticmethod
    def replace_text(p,text):
        runs=p.findall(qn('w:r'));rpr=deepcopy(runs[0].find(qn('w:rPr'))) if runs and runs[0].find(qn('w:rPr')) is not None else None
        for child in list(p):
            if child.tag!=qn('w:pPr'):p.remove(child)
        r=element('w:r')
        if rpr is not None:r.append(rpr)
        r.append(element('w:t',str(text)));p.append(r)

    def add_node(self,node):
        self.final_sect.addprevious(node)
        return node

    def p(self,text,style='Normal',bold=False):
        p=self.doc.add_paragraph(style=style)
        p.add_run(text).bold=bold if bold else None
        return p

    def h(self,text,level=2):return self.p(text,f'Heading {level}')
    def page(self):self.doc.add_page_break()

    def caption(self,kind,text):
        p=deepcopy(self.caption_sample)
        for c in list(p):
            if c.tag!=qn('w:pPr'):p.remove(c)
        if kind=='Table':self.table_no+=1;n=self.table_no
        else:self.figure_no+=1;n=self.figure_no
        for s in [kind+' ']:
            r=element('w:r');r.append(element('w:t',s));p.append(r)
        for typ in ['begin']:
            r=element('w:r');r.append(element('w:fldChar',**{'w:fldCharType':typ}));p.append(r)
        r=element('w:r');r.append(element('w:instrText',f' SEQ {kind} \\* ARABIC '));p.append(r)
        r=element('w:r');r.append(element('w:fldChar',**{'w:fldCharType':'separate'}));p.append(r)
        r=element('w:r');r.append(element('w:t',str(n)));p.append(r)
        r=element('w:r');r.append(element('w:fldChar',**{'w:fldCharType':'end'}));p.append(r)
        r=element('w:r');r.append(element('w:t',': '+text));p.append(r)
        self.add_node(p)
        return n

    @staticmethod
    def fill_table(tbl,headers,rows):
        samples=tbl.findall(qn('w:tr'))
        header=deepcopy(samples[0]);body=deepcopy(samples[1] if len(samples)>1 else samples[0])
        for tr in samples:tbl.remove(tr)
        for i,values in enumerate([headers]+list(rows)):
            tr=deepcopy(header if i==0 else body)
            cells=tr.findall(qn('w:tc'))
            while len(cells)<len(values):tr.append(deepcopy(cells[-1]));cells=tr.findall(qn('w:tc'))
            for cell in cells[len(values):]:tr.remove(cell)
            for cell,value in zip(cells,values):
                paras=cell.findall(qn('w:p'));p=paras[0] if paras else element('w:p')
                for c in list(cell):
                    if c.tag!=qn('w:tcPr'):cell.remove(c)
                cell.append(p);Builder.replace_text(p,str(value))
            pr=tr.find(qn('w:trPr'))
            if pr is None:pr=element('w:trPr');tr.insert(0,pr)
            if pr.find(qn('w:cantSplit')) is None:pr.append(element('w:cantSplit'))
            if i==0 and pr.find(qn('w:tblHeader')) is None:pr.append(element('w:tblHeader'))
            # Fixed-height template diagram rows must never constrain text tables.
            for h in list(pr.findall(qn('w:trHeight'))):pr.remove(h)
            tbl.append(tr)

    def table(self,headers,rows,caption,sample=4,best_rows=()):
        self.caption('Table',caption)
        tbl=deepcopy(self.table_samples[sample])
        self.fill_table(tbl,headers,rows)
        for index in best_rows:
            tr=tbl.findall(qn('w:tr'))[index+1]
            for cell in tr.findall(qn('w:tc'))[1:]:
                for run in cell.findall('.//w:r',NS):
                    pr=run.find(qn('w:rPr'))
                    if pr is None:pr=element('w:rPr');run.insert(0,pr)
                    if pr.find(qn('w:b')) is None:pr.append(element('w:b'))
        # Use source table style, replacing only grid widths for requested columns.
        grid=tbl.find(qn('w:tblGrid'))
        if grid is not None:
            for c in list(grid):grid.remove(c)
            for _ in headers:grid.append(element('w:gridCol',**{'w:w':str(9070//len(headers))}))
        for tr in tbl.findall(qn('w:tr')):
            for tc in tr.findall(qn('w:tc')):
                width=tc.find('w:tcPr/w:tcW',NS)
                if width is not None:width.set(qn('w:w'),str(9070//len(headers)))
        self.add_node(tbl)

    def image(self,name,width,caption):
        p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next=True
        p.add_run().add_picture(str(FIG/name),width=Inches(width))
        self.caption('Figure',caption)

    def equation(self,name):
        self.eq+=1;p=deepcopy(self.equation_sample)
        for child in list(p):
            if child.tag!=qn('w:pPr'):p.remove(child)
        # Source paragraph has center and right tabs; keep its native equation layout.
        r=element('w:r');r.append(element('w:tab'));p.append(r)
        math=element('m:oMath')
        for node in equation_nodes(name):math.append(node)
        p.append(math)
        r=element('w:r');r.append(element('w:tab'));r.append(element('w:t',f'({self.eq})'));p.append(r)
        self.add_node(p)

    def front(self,abstract):
        self.p('Abstract','Front Heading');self.p(abstract);self.p('Keywords: waste classification, convolutional neural networks, robustness, Streamlit, reproducibility')
        self.page();self.p('Table of Contents','Front Heading');self.add_node(self.toc);self.add_node(self.front_break)

    def save(self,out):
        # Patch the selected source header and set Word's field refresh flag.
        for section in self.doc.sections:
            for p in section.header.paragraphs:
                for r in p.runs:
                    if '[PREFIX]' in r.text:r.text=r.text.replace('[PREFIX]',self.prefix)
        settings=self.doc.settings.element
        flag=settings.find(qn('w:updateFields'))
        if flag is None:flag=element('w:updateFields');settings.append(flag)
        flag.set(qn('w:val'),'true')
        raw=io.BytesIO();self.doc.save(raw)
        original=zipfile.ZipFile(self.template);generated=zipfile.ZipFile(raw)
        replacements={'word/document.xml','word/header3.xml','word/settings.xml','word/_rels/document.xml.rels','[Content_Types].xml'}
        out.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            for item in original.infolist():
                z.writestr(deepcopy(item),generated.read(item.filename) if item.filename in replacements else original.read(item.filename))
            for name in generated.namelist():
                if name not in original.namelist():z.writestr(name,generated.read(name))
        preserved={n:original.read(n)==zipfile.ZipFile(out).read(n) for n in original.namelist() if n not in replacements}
        if not all(preserved.values()):raise RuntimeError('Template preserve-only parts changed')
        fidelity=ROOT/'tmp/submission/template_fidelity.json'
        fidelity.parent.mkdir(parents=True,exist_ok=True)
        fidelity.write_text(json.dumps({'template_sha256':hashlib.sha256(self.template.read_bytes()).hexdigest(),'unchanged_original_parts':len(preserved),'all_preserved':all(preserved.values()),'edited_parts':sorted(replacements),'equations':self.eq,'tables':self.table_no,'figures':self.figure_no},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--team',type=Path,default=ROOT/'submission/team.json')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    team=json.loads(args.team.read_text(encoding='utf-8-sig'))
    make_figures();builder=Builder(args.template,team)
    from submission_report_content import write_report
    write_report(builder)
    out=args.out or ROOT/'submission'/f'{builder.prefix}_report.docx'
    builder.save(out)
    print(out)

if __name__=='__main__':main()
