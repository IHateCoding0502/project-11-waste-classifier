"""Protect course submission gates and the clean-code publication boundary."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

ROOT=Path(__file__).resolve().parent.parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

package=load('package_submission')
audit=load('check_submission_repo')

def confirmed_team():
    # Synthetic fixture identities, not the actual team's agreement.
    return {'class_code':'61FIT3AIN-TUT02','group':5,'project':11,
            'members':[{'name':'Fixture A','student_id':'TEST-A','tasks':'Fixture task A','percent':40,'representative':True},
                       {'name':'Fixture B','student_id':'TEST-B','tasks':'Fixture task B','percent':60,'representative':False}],
            'contributions_agreed':True,'midterm_slides_confirmed':True}

def test_submission_requires_agreement_and_original_deck_confirmation():
    team=confirmed_team();assert package.readiness(team)==[]
    team['midterm_slides_confirmed']=False
    assert any('midterm' in issue for issue in package.readiness(team))
    team=confirmed_team();team['contributions_agreed']=False
    assert any('agreement' in issue.lower() for issue in package.readiness(team))

def test_percentages_must_be_integers_sum_to_100_and_have_one_representative():
    for value in [None,40.0,True,-1,101]:
        team=confirmed_team();team['members'][0]['percent']=value
        assert any('integer' in issue for issue in package.readiness(team))
    team=confirmed_team();team['members'][0]['percent']=39
    assert package.readiness(team)
    for members in [[],[dict(m,representative=False) for m in confirmed_team()['members']],
                    [dict(m,representative=True) for m in confirmed_team()['members']]]:
        team=confirmed_team();team['members']=members
        assert package.readiness(team)

def test_publication_audit_excludes_forbidden_files_but_allows_team_checkpoint():
    bad=['.vscode/settings.json','.env','.env.production','data/train/plastic/a.jpg',
         'data/validation/a.jpg','data/test/a.jpg','env/bin/python','cache/__pycache__/x.pyc',
         'models/downloaded.pt','submission/team.json','kaggle.json']
    assert len(audit.prohibited_paths(bad))==len(bad)
    assert audit.prohibited_paths(['models/waste_cnn.pth','data/sample/paper/a.png',
                                   'submission/team.example.json','.env.example'])==[]

def test_group_prefix_and_review_notice_are_explicit():
    team=confirmed_team()
    assert package.prefix_of(team)=='61FIT3AIN-TUT02_G05_P11'
    text=package.code_readme(package.prefix_of(team),['Unconfirmed original midterm PDF'])
    assert text.startswith('# REVIEW PACKAGE NOT READY FOR SUBMISSION')
    assert 'Unconfirmed original midterm PDF' in text
    assert 'python.exe -m streamlit run app.py' in text

def test_stale_appendix_cannot_be_marked_ready_by_changing_metadata():
    ns={'w':'urn:word-test'}
    doc=ET.Element('{urn:word-test}document');table=ET.SubElement(doc,'{urn:word-test}tbl')
    team=confirmed_team()
    rows=[['Student ID','Full name','Main tasks','%']]+[
        [m['student_id'],m['name']+('*' if m['representative'] else ''),m['tasks'],str(m['percent'])]
        for m in team['members']]
    for values in rows:
        row=ET.SubElement(table,'{urn:word-test}tr')
        for value in values:
            cell=ET.SubElement(row,'{urn:word-test}tc')
            ET.SubElement(cell,'{urn:word-test}t').text=value
    package.validate_contributions(doc,ns,team)
    team['members'][0]['tasks']='Changed after report verification'
    with pytest.raises(ValueError,match='Appendix B differs'):
        package.validate_contributions(doc,ns,team)
