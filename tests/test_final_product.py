"""Upload safety, preprocessing equivalence, real checkpoint and product tests."""
import io
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from starter.inference import decode_upload,infer,MAX_BYTES
from starter.student_core import CLASSES,build_model,load_trained_model,image_to_tensor,count_parameters
from scripts.evaluate_stress_tests import read_manifest,apply_transform,bootstrap

torch.set_num_threads(2)

def encoded(mode='RGB',size=(31,15),fmt='PNG'):
    stream=io.BytesIO()
    Image.new(mode,size).save(stream,format=fmt)
    return stream.getvalue()

@pytest.mark.parametrize('mode,fmt', [('RGB','JPEG'),('RGB','PNG'),('L','PNG'),('RGBA','PNG'),('P','PNG')])
def test_upload_formats(mode,fmt):
    upload=decode_upload(encoded(mode,fmt=fmt))
    assert upload.image.mode=='RGB' and upload.image.size==(31,15)
    assert upload.format==fmt

def test_transparent_background_and_exif():
    assert decode_upload(encoded('RGBA')).image.getpixel((0,0))==(255,255,255)
    stream=io.BytesIO(); exif=Image.Exif(); exif[274]=6
    Image.new('RGB',(10,20)).save(stream,format='JPEG',exif=exif)
    upload=decode_upload(stream.getvalue())
    assert upload.original_size==(10,20) and upload.image.size==(20,10)

@pytest.mark.parametrize('data',[b'',b'corrupted',encoded(fmt='GIF'),encoded()[:30],b'x'*(MAX_BYTES+1),encoded(size=(8193,1)),encoded(size=(5000,5000))], ids=['empty','corrupt','gif','truncated','byte_limit','side_limit','pixel_limit'])
def test_reject_unsafe_images(data):
    with pytest.raises(ValueError): decode_upload(data)

@pytest.mark.parametrize('size',[(1,300),(500,1),(60,60)])
def test_aspect_ratios_and_training_preprocessing(size):
    data=encoded(size=size)
    upload=decode_upload(data)
    model=build_model(augmentation='none'); model.eval()
    tensor=image_to_tensor(io.BytesIO(data),(64,64)).unsqueeze(0)
    with torch.inference_mode(): expected=torch.softmax(model(tensor),1)[0].numpy()
    result=infer(model,upload.image)
    np.testing.assert_allclose(list(result['scores'].values()),expected,rtol=1e-5)
    assert result['seconds']>=0 and list(result['scores'])==list(CLASSES)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_existing_selected_checkpoint(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    path=ROOT/'models/waste_cnn.pth'
    if not path.exists(): pytest.skip('External selected checkpoint not supplied')
    model=load_trained_model(path,device=device)
    assert count_parameters(model)==114966 and model.dropout_p==.3 and model.augmentation=='none'
    result=infer(model,Image.new('RGB',(64,64)))
    assert sum(result['scores'].values())==pytest.approx(1,abs=1e-6)

def test_manifest_rejects_invalid_labels_and_duplicate_objects(tmp_path):
    path=tmp_path/'bad.csv'
    path.write_text('path,true_label,condition,physical_object_id\ndata/sample/paper/paper_01.png,unknown,campus_novel,1\n')
    with pytest.raises(ValueError): read_manifest(path)
    path.write_text('path,true_label,condition,physical_object_id,capture_device\ndata/sample/paper/paper_01.png,paper,campus_novel,1,phone\n')
    with pytest.raises(ValueError): read_manifest(path)
    assert 'status' in bootstrap([{'physical_object_id':'one','true_label':'paper'}])

def test_guidance_schema_complete():
    guidance=json.loads((ROOT/'data/disposal_guidance.json').read_text())
    assert set(guidance)==set(CLASSES)
    for entry in guidance.values():
        assert all(entry.get(f) for f in ['category','description','examples','preparation','general_advice','warnings','local_guidance','sources'])
        assert all(s['url'].startswith('https://') for s in entry['sources'])

@pytest.mark.parametrize("entry", ["app.py", "starter/app.py"])
def test_app_pages(entry):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / entry)).run(timeout=30)
    assert not app.exception and app.button(key="start_classifying")
    for page in ("Waste Classification", "Model Comparison", "Model Analytics",
                 "Waste Sorting Guide", "About Project"):
        app.sidebar.radio(key="navigation").set_value(page).run(timeout=30)
        assert not app.exception and app.title[0].value == page


def test_app_upload_preview_prediction_and_error(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    class Upload:
        name = "real_upload.png"
        data = encoded()

        def getvalue(self):
            return self.data

    upload = Upload()
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    app.button(key="start_classifying").click().run(timeout=30)
    assert not app.exception and any("Original 31" in c.value for c in app.caption)
    next(button for button in app.button if button.label == "Classify Waste").click().run(timeout=30)
    assert not app.exception and len(app.get("progress")) == 6 and len(app.metric) == 2
    result = next(iter(app.session_state["cnn_results"].values()))
    assert app.metric[0].value == f"{result['confidence']:.1%}"
    assert any(result['label'].title() in m.value for m in app.markdown)
    assert any(m.value == f"Sorting {result['label']}" for m in app.subheader)
    # An unrelated rerun must keep both the displayed result and the session count.
    app.run(timeout=30)
    assert len(app.get("progress")) == 6 and app.session_state["cnn_runs"] == 1
    upload.data = encoded(size=(32, 16))
    app.run(timeout=30)
    assert not app.exception and not app.metric and not app.get("progress")
    upload.data = b"invalid"
    app.run(timeout=30)
    assert not app.exception and len(app.error) == 1
