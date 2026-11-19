"""Audit tracked files against AIN guide section 4.2 without reading secrets.

Run from a Git checkout. Local ignored data/environments are intentionally allowed;
the check covers what a clean clone or git archive would actually contain.
"""
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent
FORBIDDEN_DIRS={'.git','.venv','venv','env','.venv-submission','.venv-qa',
                '__pycache__','.pytest_cache','.ipynb_checkpoints','.idea','.vscode','__MACOSX'}
REQUIRED={'README.md','requirements.txt','app.py','starter/student_core.py',
          'starter/app.py','starter/inference.py','models/waste_cnn.pth',
          'scripts/run_experiments.py','scripts/build_analysis_figures.py',
          'scripts/package_submission.py','submission/README.md',
          'submission/team.example.json','tests/test_project_11_sanity.py'}

def prohibited_paths(paths):
    issues=[]
    for name in paths:
        p=PurePosixPath(name.replace('\\','/'));parts=set(p.parts);base=p.name.lower()
        if parts & FORBIDDEN_DIRS or any(part.startswith('.venv') for part in p.parts):
            issues.append(f'Environment, editor or cache file: {name}')
        elif name.startswith(('data/train/','data/validation/','data/test/')) or base=='dataset-resized.zip':
            issues.append(f'Full dataset: {name}')
        elif base in {'.env','kaggle.json','.ds_store','team.json'} or (base.startswith('.env.') and base!='.env.example') or p.suffix.lower() in {'.pem','.key','.pyc'}:
            issues.append(f'Credential, private metadata or generated file: {name}')
        elif p.suffix.lower() in {'.h5','.pt','.pth','.pkl','.ckpt'} and name!='models/waste_cnn.pth':
            issues.append(f'Unapproved model artifact: {name}')
    return issues

def main():
    result=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,check=True)
    paths=[p.decode('utf-8') for p in result.stdout.split(b'\0') if p]
    issues=prohibited_paths(paths)
    issues += [f'Required source file is not tracked: {p}' for p in sorted(REQUIRED-set(paths))]
    if issues:
        print('\n'.join(issues));return 1
    print(f'PASS: {len(paths)} tracked files; required starter, checkpoint and packaging tools present; no prohibited paths.')
    print('Course readiness still requires the local DOCX, confirmed original slides and agreed contributions.')
    return 0

if __name__=='__main__':sys.exit(main())
