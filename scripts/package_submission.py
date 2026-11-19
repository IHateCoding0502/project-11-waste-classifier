"""Create and verify the AIN submission ZIP without virtualenvs or full datasets.
Default mode refuses incomplete member agreement or unconfirmed midterm slides.
--review builds an explicitly incomplete review package for local checking.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent.parent
VALID_CLASSES={'61FIT3AIN-TUT01','61FIT3AIN-TUT02','66FIT3AIN-TT02','66FIT3AIN-TT03','62FIT4ATI-CLC02','62FIT4ATI-CLC03'}
PATTERNS=['app.py','README.md','.gitignore','requirements*.txt','run_app.ps1','.streamlit/*.toml','assets/*.css',
          'starter/*.py','scripts/*.py','scripts/*.ps1','tests/*.py','data/sample/**/*',
          'data/LICENSE.md','data/README.md','data/disposal_guidance.json',
          'data/stress_tests/*.csv','data/stress_tests/*.json','data/stress_tests/*.md',
          'data/stress_tests/**/.gitkeep','models/waste_cnn.pth',
          'results/*.json','results/*.csv','results/*.md','results/figures/*.png',
          'results/stress/*.json','results/stress/*.csv','results/stress/*.png',
          'reports/architecture.png','reports/ui-desktop.jpg','reports/ui-mobile.jpg',
          'results/compatibility311/metrics.json','results/compatibility311/scenario_comparison.csv',
          'ORAL_DEFENSE.md','DEMO_SCRIPT.md','FINAL_DELIVERY_CHECKLIST.md']
FORBIDDEN={'.git','.venv','venv','env','__pycache__','.pytest_cache','.ipynb_checkpoints',
           '.idea','.vscode','__MACOSX','.venv-submission','.venv-qa'}


def prefix_of(team):
    if team['class_code'] not in VALID_CLASSES:raise ValueError('Class code is not listed in the guide.')
    if not 1<=team['group']<=99 or team['project']!=11:raise ValueError('Invalid group or P11 project code.')
    return f"{team['class_code']}_G{team['group']:02d}_P{team['project']:02d}"


def readiness(team):
    issues=[];members=team.get('members',[])
    if not members or any(not m.get('name') or not m.get('student_id') for m in members):issues.append('Member names or student IDs missing.')
    if sum(bool(m.get('representative')) for m in members)!=1:issues.append('Exactly one representative is required.')
    values=[m.get('percent') for m in members]
    if not values or any(type(v) is not int or not 0<=v<=100 for v in values) or sum(v for v in values if type(v) is int)!=100:issues.append('Agreed integer contribution percentages must total 100.')
    if any(not m.get('tasks','').strip() for m in members):issues.append('Member task descriptions are missing.')
    if not team.get('contributions_agreed'):issues.append('Group agreement on contribution table is unconfirmed.')
    if not team.get('midterm_slides_confirmed'):issues.append('The exact previously presented midterm PDF is unconfirmed.')
    return issues


def validate_contributions(doc,ns,team):
    def text(node):return ' '.join(''.join(n.text or '' for n in node.findall('.//w:t',ns)).split())
    for table in doc.findall('.//w:tbl',ns):
        rows=[[text(cell) for cell in row.findall('w:tc',ns)] for row in table.findall('w:tr',ns)]
        if rows and rows[0]==['Student ID','Full name','Main tasks','%']:
            expected=[[m['student_id'],m['name']+('*' if m['representative'] else ''),
                       m['tasks'] or 'Awaiting group confirmation',
                       str(m['percent']) if m['percent'] is not None else 'Pending'] for m in team['members']]
            expected=[[' '.join(str(v).split()) for v in row] for row in expected]
            if rows[1:]!=expected:raise ValueError('Appendix B differs from team.json; rebuild and reverify the Word report.')
            return
    raise ValueError('Appendix B member contribution table is missing.')


def validate_report(path,team=None):
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    with zipfile.ZipFile(path) as z:
        doc=ET.fromstring(z.read('word/document.xml'));text=' '.join(n.text or '' for n in doc.findall('.//w:t',ns))
        if team is not None:validate_contributions(doc,ns,team)
        for marker in ['Template Instructions','[PROJECT TITLE]','[PREFIX]','[Student ID]','[Contribution 1]','[Write the abstract here','Recommended length:']:
            if marker in text:raise ValueError(f'Template instruction/placeholder remains: {marker}')
        if len(doc.findall('.//m:oMath',ns))<10:raise ValueError('Expected native Word equations are missing.')
        instructions=' '.join(n.text or '' for n in doc.findall('.//w:instrText',ns))
        if 'TOC' not in instructions or 'SEQ Table' not in instructions or 'SEQ Figure' not in instructions:raise ValueError('Live TOC/caption fields missing.')
    verification=ROOT/'submission/report_verification.json'
    if not verification.exists():raise ValueError('Report pagination has not been verified.')
    record=json.loads(verification.read_text(encoding='utf-8'))
    if record.get('body_pages')!=15:raise ValueError('The main body must be exactly 15 pages.')
    if record.get('docx_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Report changed after pagination verification.')


def code_readme(prefix,issues):
    notice=('# REVIEW PACKAGE NOT READY FOR SUBMISSION\n\n'+''.join('- '+i+'\n' for i in issues)+'\n') if issues else ''
    return notice+f'''# Visual Waste Sorting Assistant

AIN Project P11. Python **3.11**. Original PyTorch CNN with 114,966 trainable parameters.
GitHub: https://github.com/IHateCoding0502/project-11-waste-classifier

## Install

Windows PowerShell, from this code directory:

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

Linux/macOS: `python3.11 -m venv .venv`, activate it, then use the same pip commands.
NumPy 2.3.5 is selected for Python 3.11; Python 3.12+ uses the historical 2.5.4 pin.
Initial dependency installation needs a package source. After installation, inference works offline.
No hosted LLM API replaces the AI core. The supplied checkpoint was produced by the team's
training script and loads on CPU automatically; CUDA requires a compatible PyTorch build.

## One command to run

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

The legacy entry point `streamlit run starter/app.py` also works. Open the displayed localhost URL,
upload one JPEG/PNG, inspect its preview and select Predict. All six scores and material guidance
are displayed. The four pages preserve model details, failure analysis and saved stress results.

## Tests and model validation

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/validate_checkpoint.py
```

Independent Python 3.11.16 CPU verification: 38 passed, one CUDA test skipped. Full test
images are intentionally absent from this archive; the included synthetic samples exercise
software behavior only. CPU/CUDA model tests use the supplied team-trained checkpoint.

## Reproduce experiments and report figures

```powershell
python scripts/prepare_dataset.py --confirm
python scripts/run_experiments.py --epochs 25 --seed 42
python scripts/build_analysis_figures.py
python scripts/evaluate_stress_tests.py --prepare --device cpu
```

Dataset acquisition is an optional preparation step, not a runtime inference requirement.
See data/LICENSE.md for the pinned source and checksum. Training and evaluation run locally
after data preparation. The guide excludes data/train, data/validation and data/test from the ZIP.
Saved results remain in results/ so numeric tables and galleries can be inspected offline.
Appendix A maps commands to tables and figures. Original baselines and ablations use seed 42;
retraining is not guaranteed to reproduce numbers bit-for-bit. Never alter new scores to match old ones.

## Checkpoint and preprocessing

models/waste_cnn.pth is the selected aug_none model: dropout .3, no augmentation,
RGB/bilinear 64×64, float32 /255, CHW, no mean/std normalization.
Class indices: cardboard=0, glass=1, metal=2, paper=3, plastic=4, trash=5.
SHA-256: 825083ce60c04c0566b736a2089ece09f9a640ba0b72b55ce3b1fadf6ef077c3.

## Report and slides

The required Word report is `../report/{prefix}_report.docx`.
The original midterm PDF is `../slides/{prefix}_midterm_slides.pdf`; no final slides are required.
The report contains exactly 15 main-body pages. Source report scripts are included; rebuilding
the DOCX needs requirements-report.txt, the original course template and member metadata.
Microsoft Word on Windows can update fields and render via scripts/render_submission_word.ps1.

## Limits and missing evidence

Original CNN: 334/435 correct, 76.8% accuracy, 75.7% Macro-F1. Controlled synthetic stress:
1,740 images, 48.6% accuracy, 48.4% Macro-F1. The 12-image complex subset overlaps standard
data. Real-camera low-light/blur and novel campus evaluations remain pending genuine photographs.
Follow data/stress_tests/COLLECTION_CHECKLIST.md. Softmax is uncalibrated; unknown objects
still receive a known class. High confidence does not guarantee correctness or recycling acceptance.
Disposal guidance distinguishes general preparation from actual local collection arrangements.
'''


def create(team,review=False,outdir=None):
    prefix=prefix_of(team);issues=readiness(team)
    if issues and not review:raise ValueError('Submission is not ready:\n- '+'\n- '.join(issues))
    report=ROOT/'submission'/f'{prefix}_report.docx';validate_report(report,team)
    slides=(ROOT/team['midterm_slides']).resolve()
    if not slides.is_file() or slides.suffix.lower()!='.pdf':raise ValueError('Midterm PDF not found.')
    files=set()
    for pattern in PATTERNS:files.update(p for p in ROOT.glob(pattern) if p.is_file())
    outdir=Path(outdir or ROOT/'submission');outdir.mkdir(parents=True,exist_ok=True)
    path=outdir/f'{prefix}.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr(prefix+'/',b'')
        for folder in ['slides','code','report']:z.writestr(f'{prefix}/{folder}/',b'')
        z.write(slides,f'{prefix}/slides/{prefix}_midterm_slides.pdf')
        z.write(report,f'{prefix}/report/{prefix}_report.docx')
        for p in sorted(files):
            rel=p.relative_to(ROOT)
            if set(rel.parts)&FORBIDDEN or p.suffix.lower() in {'.pyc','.tmp','.log'} or '.env' in rel.parts:raise ValueError(f'Unsafe package candidate: {rel}')
            if rel.as_posix().startswith(('data/train/','data/test/','data/validation/')):raise ValueError('Full dataset must be excluded.')
            if rel.as_posix()=='README.md':
                z.writestr(f'{prefix}/code/README.md',code_readme(prefix,issues))
            else:z.write(p,f'{prefix}/code/{rel.as_posix()}')
        # Local metadata accompanies the report generator; it is not a credential.
        z.writestr(f'{prefix}/code/submission/team.json',json.dumps(team,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if z.testzip():raise ValueError('ZIP integrity check failed.')
        if {n.split('/')[0] for n in names}!={prefix}:raise ValueError('ZIP has multiple roots.')
        if {n.split('/')[1] for n in names if len(n.split('/'))>2}!={'slides','code','report'}:raise ValueError('Incorrect top-level folders.')
        for n in names:
            if set(Path(n).parts)&FORBIDDEN:raise ValueError(f'Forbidden ZIP member: {n}')
            if any('/data/'+s+'/' in n for s in ['train','validation','test']):raise ValueError('Full data leaked into ZIP.')
        if hashlib.sha256(z.read(f'{prefix}/slides/{prefix}_midterm_slides.pdf')).hexdigest()!=hashlib.sha256(slides.read_bytes()).hexdigest():raise ValueError('Original slides were altered.')
        # Only the original midterm deck is a PDF in the submission.
        if len([n for n in names if n.lower().endswith('.pdf')])!=1:raise ValueError('Extra PDF in submission.')
    if path.stat().st_size>=50*1024*1024:raise ValueError('ZIP exceeds recommended 50 MiB size.')
    result={'zip':str(path),'prefix':prefix,'ready_to_submit':not issues,'blocking_items':issues,
            'file_count':len(names),'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'report_body_pages':15,'slides_unchanged':True,'zip_structure_verified':True}
    (outdir/'package_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--team',type=Path,default=ROOT/'submission/team.json');p.add_argument('--review',action='store_true');p.add_argument('--outdir',type=Path)
    a=p.parse_args();create(json.loads(a.team.read_text(encoding='utf-8-sig')),a.review,a.outdir)
