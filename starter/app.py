"""Eco-Tech Streamlit UI. The original CNN and shared inference remain unchanged."""
from __future__ import annotations
import csv
import hashlib
import html
import json
import sys
from pathlib import Path
import streamlit as st
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from starter.student_core import CLASSES, load_trained_model
from starter.inference import decode_upload, infer

WARNING = ('This model is an educational prototype. Predictions may be incorrect, '
           'especially for visually similar materials or images outside the training dataset.')
PAGES = ['Waste Classifier', 'Model Information', 'Known Limitations & Failure Modes', 'Robustness Testing']
ICONS = {
 'leaf':'<path d="M20 4c-8-1-16 2-15 9 1 6 9 7 12 2 2-3 2-7 3-11Z"/><path d="m4 21 10-11M8 16v-5m1 4h5"/>',
 'scan':'<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M7 12h10"/>',
 'grid':'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
 'chip':'<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 9h6v6H9zM9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/>',
 'upload':'<path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5"/>',
 'image':'<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1"/><path d="m3 17 5-5 4 4 4-6 5 7"/>',
 'cardboard':'<path d="m12 3 9 5v9l-9 5-9-5V8l9-5Zm0 10 9-5M3 8l9 5v9M7 5.5l9 5v5"/>',
 'glass':'<path d="M9 3h6v7l4 10H5l4-10V3Zm0 4h6M7 15h10"/>',
 'metal':'<ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v14c0 4 14 4 14 0V5M5 9c0 4 14 4 14 0M5 16c0 4 14 4 14 0"/>',
 'paper':'<path d="M6 3h9l4 4v14H6V3Zm9 0v5h4M9 12h7m-7 4h7"/>',
 'plastic':'<path d="M10 3h4v5l3 4v8H7v-8l3-4V3Zm0 3h4M7 14h10M7 17h10"/>',
 'trash':'<path d="M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7"/>',
 'check':'<path d="m5 12 4 4L19 6"/><circle cx="12" cy="12" r="10"/>',
 'alert':'<path d="m12 3 10 18H2L12 3Zm0 6v5m0 3v1"/>',
 'chart':'<path d="M4 3v18h18M8 16v-5m5 5V7m5 9V4"/>',
}

def icon(name):
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS.get(name,ICONS["leaf"])}</svg>'

def markup(value):
    st.markdown(value, unsafe_allow_html=True)

def escape(value):
    return html.escape(str(value))

@st.cache_resource(show_spinner='Loading original PyTorch CNN…')
def get_model(path: str, mtime_ns: int):
    return load_trained_model(path)

def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def section(title, note='', symbol='leaf'):
    markup(f'<div class="section-label">{icon(symbol)}{escape(title)}</div><p class="section-note">{escape(note)}</p>')

def header(title, subtitle, badge='Original PyTorch CNN', pending=False):
    markup(f'<div class="page-head"><div><div class="eyebrow">Visual Waste AI / Research workspace</div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div><div class="badge {"amber" if pending else ""}"><span class="dot {"pending" if pending else ""}"></span>{escape(badge)}</div></div>')

def kpis(items):
    cards=''.join(f'<div class="kpi"><div class="icon-tile">{icon(symbol)}</div><div><strong>{escape(value)}</strong><small>{escape(label)}</small></div></div>' for value,label,symbol in items)
    markup(f'<div class="kpi-grid {"four" if len(items)==4 else ""}">{cards}</div>')

def render_guidance(label, key='guidance'):
    item=read_json(ROOT/'data/disposal_guidance.json')[label]
    with st.container(key=key):
        markup(f'<div class="guide-top"><div class="icon-tile">{icon(label)}</div><div><div class="eyebrow">Sort with care</div><h3>Disposal guidance · {escape(item["category"])}</h3><p>{escape(item["description"])}</p></div></div>')
        cells=[]
        for field,title in [('examples','Common examples'),('preparation','Before disposal'),('general_advice','General guidance'),('warnings','Safety & acceptance')]:
            cells.append(f'<div class="guide-cell {"warning" if field=="warnings" else ""}"><small>{title}</small><p>{escape(item[field])}</p></div>')
        markup('<div class="guide-grid">'+''.join(cells)+'</div>')
        with st.expander('Local collection guidance & sources'):
            st.info(item['local_guidance'])
            for source in item['sources']:
                st.markdown(f'[{source["title"]}]({source["url"]})')
            st.caption(f'Reviewed {item["reviewed"]}. Facility acceptance is not guaranteed.')

def result_summary(result):
    markup('<span class="badge">Analysis complete · verify the material</span>')
    markup(f'<div class="prediction-label">PREDICTED MATERIAL</div><div class="prediction-name">{icon(result["label"])}{escape(result["label"].title())}</div>')
    a,b=st.columns(2)
    a.metric('Model confidence score',f'{result["confidence"]:.1%}')
    b.metric('Inference time',f'{result["seconds"]*1000:.2f} ms')
    st.caption('Model confidence score — not calibrated probability.')
    st.warning('Confidence scores are not calibrated probabilities. Predictions may be incorrect.')
    st.caption(f'Device: {result["device"]} · Forward + softmax only; excludes loading and preprocessing.')

def confidence_analysis(result):
    with st.container(key='confidence'):
        section('Class Confidence Analysis','All six scores from the original CNN. The leading score is highlighted.','chart')
        for label in CLASSES:
            value=result['scores'][label]
            winner=label==result['label']
            markup(f'<div class="confidence-label {"winner" if winner else ""}"><span>{label.title()}{"<span class=top>TOP SCORE</span>" if winner else ""}</span><span>{value:.2%}</span></div>')
            st.progress(value)
    st.caption('Unknown objects still receive one of the six labels. Confidence is not a validated unknown-object detector.')
    st.caption(WARNING)

def render_prediction(model,image):
    # Kept as an entry point for existing integrations. Shared prediction is unchanged.
    result=infer(model,image)
    result_summary(result)
    confidence_analysis(result)
    render_guidance(result['label'])


def classifier_page(model,model_error,path):
    header('Waste Classification Studio','AI-powered visual waste identification. One image, six material categories.',
           'Original PyTorch CNN loaded' if model is not None else 'Model unavailable',model is None)
    kpis([('6','Supported Categories','grid'),('64 × 64','Input Resolution','scan'),('Original CNN','PyTorch Model','chip')])
    if model_error:
        st.error(model_error)
    with st.container(key='upload'):
        section('Upload a waste image','Drag and drop a clear photo of a single waste item, or browse your files.','upload')
        uploaded=st.file_uploader('Upload a waste photo',type=['jpg','jpeg','png'],label_visibility='collapsed',key='waste_upload')
        markup('<div class="upload-foot">JPG / JPEG / PNG · Up to 10 MiB · Maximum 20 megapixels and 8192 px per side · EXIF orientation supported</div>')
    if uploaded is None:
        left,right=st.columns([1,1])
        with left,st.container(key='preview'):
            section('Image preview','Your original image will appear here.','image')
            markup(f'<div class="empty-preview"><div class="icon-tile">{icon("scan")}</div><strong>A clearer view. A better starting point.</strong><p>Upload an image to inspect its orientation, dimensions and format before classification.</p></div>')
        with right,st.container(key='result'):
            section('Ready when you are','A simple workflow for thoughtful waste sorting.','leaf')
            markup('<div class="workflow"><span>1</span> Upload <span>2</span> Classify <span>3</span> Review & sort</div>')
            st.write('Start with one visible item in good lighting. Review the prediction, compare the material and check the disposal guidance.')
            st.info('Experimental six-class classifier. Every image receives a known label, including unsupported objects.')
            st.caption('Confidence is a model score, not a guarantee of correct identification.')
        with st.expander('Explore disposal guidance for all six materials'):
            render_guidance(st.selectbox('Material',CLASSES,format_func=str.title))
        return
    try:
        raw=uploaded.getvalue()
        upload=decode_upload(raw)
        identity=(hashlib.sha256(raw).hexdigest(),path.stat().st_mtime_ns if path.exists() else None)
        stored=st.session_state.get('classification')
        result=stored['result'] if stored and stored['identity']==identity else None
        left,right=st.columns([1,1])
        with left,st.container(key='preview'):
            section('Original image','Check the object before running classification.','image')
            st.image(upload.image,caption=uploaded.name,width='stretch')
            st.caption(f'{upload.format} · Original {upload.original_size[0]} × {upload.original_size[1]} px · Oriented {upload.image.width} × {upload.image.height} px')
        with right,st.container(key='result'):
            section('Classification result','Powered by the selected aug_none checkpoint.','chip')
            if st.button('Predict',type='primary',width='stretch',icon=':material/auto_awesome:',disabled=model is None):
                result=infer(model,upload.image)
                st.session_state['classification']={'identity':identity,'result':result}
            if model_error:
                st.error(model_error)
            elif result:
                result_summary(result)
            else:
                markup(f'<div class="empty-preview"><div class="icon-tile">{icon("leaf")}</div><strong>Your image is ready to classify</strong><p>Select Predict to see the material, all six model scores and relevant disposal guidance.</p></div>')
        if result:
            confidence_analysis(result)
            render_guidance(result['label'])
            with st.expander('Browse guidance for another material'):
                label=st.selectbox('Other material',CLASSES,format_func=str.title,index=CLASSES.index(result['label']))
                if label!=result['label']:
                    render_guidance(label,key='other_guidance')
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        st.error(f'Could not process this image or model: {exc}')


def model_page():
    header('Inside the Model','A compact network, built from scratch. Transparent about how it was selected.','Existing experimental results')
    kpis([('76.8%','Test Accuracy','check'),('75.7%','Test Macro-F1','chart'),('435','Test Samples','image'),('114,966','Trainable Parameters','chip')])
    st.caption('Existing held-out TrashNet results · 334/435 correct · Macro-F1 image-bootstrap 95% CI: 70.9%–79.9%. These are not live performance measurements.')
    left,right=st.columns([1.15,1])
    with left,st.container(key='architecture'):
        section('CNN architecture','Four convolutional blocks. No pretrained weights.','chip')
        image=ROOT/'reports/architecture.png'
        if image.exists():
            st.image(str(image),width='stretch')
        else:
            st.code('RGB → [Conv → BatchNorm → ReLU → MaxPool] × 4\n→ Global average pool → Dropout → Linear → ReLU\n→ Dropout → Linear (6 logits)')
        st.caption('Channels 3 → 16 → 32 → 64 → 128 · Dense 128 → 128 → 6 · Dropout 0.3 at both head stages.')
    with right,st.container(border=True,key='training_card'):
        section('Training recipe','The original experimental configuration.','scan')
        st.table({'Setting':['Optimizer','Base learning rate','Loss','Batch size / budget','Selected configuration','Selected epoch','Preprocessing'],
                  'Value':['Adam + ReduceLROnPlateau','0.002','Weighted Cross-Entropy','32 / 25 epochs','aug_none · dropout 0.3','24 · minimum validation loss','RGB · bilinear 64×64 · float32 / 255']})
        st.caption('No mean/std normalization. Class indices: '+str(dict(enumerate(CLASSES))))
        st.info('Model selection trade-off: aug_none validation Macro-F1 0.720; no_reg 0.815. no_reg was excluded by the dropout > 0 requirement. This does not prove no_reg overfits.')
    with st.container(border=True,key='dataset_card'):
        section('TrashNet dataset','2,527 images · Seed 42 · Original fixed image split','grid')
        kpis([('1,766','Training images','image'),('326','Validation images','image'),('435','Testing images','image')])
        st.caption('Six output classes: Cardboard, Glass, Metal, Paper, Plastic and Trash. Image-level splitting does not rule out physical-object leakage.')
    with st.container(border=True,key='benchmark_card'):
        section('Existing test benchmark','Test metrics are separate from validation model-selection metrics.','chart')
        st.table([{'Model':'Majority baseline','Test accuracy':'23.4%','Test Macro-F1':'6.3%'},
                  {'Model':'Color histogram centroid','Test accuracy':'38.4%','Test Macro-F1':'35.3%'},
                  {'Model':'Selected CNN · aug_none','Test accuracy':'76.8%','Test Macro-F1':'75.7%'}])
        selected=read_json(ROOT/'results/selected.json')
        if selected:
            with st.expander('Original metric records & uncertainty'):
                st.json({'original_test':selected['test'],'test_ci95':selected['test_ci95']})


def limitations_page():
    header('Know the Blind Spots','Understanding failure is part of using the model responsibly.','Educational prototype',True)
    st.warning(WARNING)
    kpis([('16 / 13','Glass → Metal / Metal → Glass errors','glass'),('9','Glass → Plastic errors','plastic'),('91–99.9%','Scores of 12 most confident errors','alert')])
    cards=[
        ('Similar visual materials','Glass, metal and plastic share colors, shapes and reflective surfaces. Paper and cardboard have the strongest original F1; glass, trash and metal are more difficult.'),
        ('High-confidence errors','A strong score does not guarantee a correct label. A 50% threshold missed the known highly confident mistakes. Scores are not calibrated probabilities.'),
        ('64 × 64 resolution','Every image is resized to 64 × 64 pixels. Small texture cues and material details can disappear.'),
        ('Labels & packaging','Printed labels and patterns may influence predictions. This is a possible explanation, not an experimentally established cause.'),
        ('Unknown objects & domain shift','Unsupported objects still receive one of six labels. There is no validated unknown detector. Real backgrounds, camera angles, lighting and blur may differ from TrashNet.'),
        ('Data independence & uncertainty','A single seed and possible object-level leakage limit conclusions. The image-bootstrap interval does not capture training variability. Genuine campus testing is still pending.')]
    markup('<div class="warning-grid">'+''.join(f'<div class="warning-card"><h3>{icon("alert")}{escape(title)}</h3><p>{escape(body)}</p></div>' for title,body in cards)+'</div>')
    with st.container(border=True,key='errors_card'):
        section('Confident. And still incorrect.','Highest-confidence errors on the standard held-out set.','image')
        image=ROOT/'results/stress/standard_high_confidence_errors.png'
        if image.exists():
            st.image(str(image),width='stretch')
    with st.expander('What the exploratory Qwen comparison can tell us'):
        st.write('Qwen2.5-VL 3B via Ollama: 30 images, CNN 25/30 vs Qwen 20/30, original approximate timing 0.01 vs 1.28 s/image. Sample size, prompting and hardware constrain interpretation. This does not establish general CNN superiority.')


def scenario_record(name,record):
    if record and record.get('n_samples',0):
        markup(f'<div class="scenario-stats"><div><b>{record["n_samples"]:,}</b><small>Evaluated images</small></div><div><b>{record["accuracy"]:.1%}</b><small>Accuracy</small></div><div><b>{record["macro_f1"]:.1%}</b><small>Macro-F1</small></div></div>')
        cm=record.get('confusion_matrix',[])
        if cm:
            pairs=sorted([(cm[i][j],CLASSES[i],CLASSES[j]) for i in range(6) for j in range(6) if i!=j],reverse=True)
            st.caption('Observed confusions: '+' · '.join(f'{a.title()} → {b.title()} ({n})' for n,a,b in pairs[:2] if n))
        st.caption(f'Mean forward + softmax: {record["mean_seconds"]*1000:.2f} ms/image. Excludes image loading and preprocessing.')
        with st.expander('Examples, charts & detailed metrics'):
            for suffix in ['correct','incorrect','confusion','per_class','confidence']:
                image=ROOT/f'results/stress/{name}_{suffix}.png'
                if image.exists():
                    st.image(str(image),width='stretch',caption=suffix.replace('_',' ').title())
            st.json(record,expanded=False)
    else:
        markup('<div class="pending-panel"><strong>Pending · 0 evaluated images</strong><p>No accuracy or Macro-F1 is available. Genuine photographs and independent labels are required before evaluation.</p></div>')


def robustness_page():
    header('Robustness Lab','Inspect measured performance across four visual conditions.','Saved evaluation results')
    summary=read_json(ROOT/'results/stress/metrics.json',{}).get('scenarios',{})
    scenarios=[('01','Standard In-Distribution','Held-out TrashNet images with original appearance and image quality.','standard'),
               ('02','Complex or Deformed Items','Visually selected complex items. A descriptive subset, not an independent challenge set.','complex'),
               ('03','Low-Light or Blurred Images','Controlled transformations and real-camera captures are reported separately.','low_light_blur_synthetic'),
               ('04','Novel Campus Images','Genuinely new campus photographs, labeled independently and kept out of training.','campus_novel')]
    for pair in [scenarios[:2],scenarios[2:]]:
        columns=st.columns(2)
        for column,(number,title,description,key) in zip(columns,pair):
            with column,st.container(border=True,key=f'scenario_{number}'):
                markup(f'<div class="scenario-num">SCENARIO {number}</div>')
                st.subheader(title)
                st.caption(description)
                if key=='low_light_blur_synthetic':
                    st.markdown('**Controlled synthetic transformations**')
                scenario_record(key,summary.get(key))
                if key=='complex':
                    st.info('12-image subset overlaps the standard set. It does not validate all requested deformations or cluttered backgrounds.')
                elif key=='low_light_blur_synthetic':
                    st.caption('Brightness ×0.30 · Contrast ×0.35 · Gaussian radius 2 px · Horizontal motion kernel 9 px. Four variants reuse each source image.')
                    st.markdown('**Real-camera low-light / blur**')
                    scenario_record('low_light_blur_real',summary.get('low_light_blur_real'))
    st.info('Synthetic images are paired sensitivity tests, not substitutes for phone captures or campus evaluation. Collect genuine images using data/stress_tests/COLLECTION_CHECKLIST.md.')
    with st.expander('Compare available scenarios'):
        for name in ['scenario_comparison.png','latency.png']:
            image=ROOT/'results/stress'/name
            if image.exists():
                st.image(str(image),width='stretch')
    with st.expander('Inspect an evaluated image'):
        manifest=ROOT/'data/stress_tests/manifest.csv'
        if manifest.exists():
            with manifest.open(encoding='utf-8') as f:
                rows=[r for r in csv.DictReader(f) if (ROOT/r['path']).is_file()]
            if rows:
                row=st.selectbox('Preview an evaluated test image',rows,format_func=lambda r:f'{r["condition"]} · {r["path"]}')
                st.image(str(ROOT/row['path']),width=320)
        else:
            st.caption('Evaluation manifest not available yet.')


def main():
    st.set_page_config(page_title='Visual Waste AI · Classification Studio',page_icon='♻️',layout='wide')
    markup('<style>'+(ROOT/'assets/streamlit.css').read_text(encoding='utf-8-sig')+'</style>')
    path=ROOT/'models/waste_cnn.pth'
    model,error=None,None
    try:
        if not path.exists():
            raise FileNotFoundError('Selected checkpoint missing: models/waste_cnn.pth. See README checkpoint instructions.')
        model=get_model(str(path),path.stat().st_mtime_ns)
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        error=str(exc)
    with st.sidebar:
        markup(f'<div class="brand"><div class="brand-mark">{icon("leaf")}</div><div><div class="brand-name">Visual Waste AI</div><div class="brand-sub">See the material. Sort with care.</div></div></div>')
        st.caption('An educational waste sorting assistant powered by the original PyTorch CNN.')
        markup('<div class="side-label">Workspace</div>')
        page=st.radio('Explore',PAGES,label_visibility='collapsed',key='navigation')
        markup('<div class="side-label">Supported materials</div><div class="material-tags">'+''.join(f'<span class="material-tag">{c.title()}</span>' for c in CLASSES)+'</div>')
        markup(f'<div class="side-status"><span class="dot {"pending" if model is None else ""}"></span>{"Model unavailable" if model is None else "Original CNN loaded"}<small>aug_none · 6 classes · 64 × 64 input<br>Experimental educational prototype</small></div>')
    if page==PAGES[0]: classifier_page(model,error,path)
    elif page==PAGES[1]: model_page()
    elif page==PAGES[2]: limitations_page()
    else: robustness_page()
    markup('<div class="app-footer"><span>Visual Waste AI · Project 11</span><span>Original research. Transparent limitations. Thoughtful sorting.</span></div>')

if __name__=='__main__':
    main()
