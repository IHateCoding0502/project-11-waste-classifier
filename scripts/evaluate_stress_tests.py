"""Manifest-driven evaluation. Never trains or mutates held-out source images."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from starter.student_core import CLASSES, load_trained_model
from starter.inference import infer
from scripts.run_experiments import metrics_from_cm, confusion_matrix

CONDITIONS = ('standard', 'complex', 'low_light_blur_synthetic', 'low_light_blur_real', 'campus_novel')
FIELDS = ['path', 'true_label', 'condition', 'physical_object_id', 'lighting', 'blur_level',
          'capture_device', 'notes', 'transform', 'source_path', 'sha256']
TRANSFORMS = {
    'dark': {'brightness': 0.3},
    'low_contrast': {'contrast': 0.35},
    'defocus': {'gaussian_radius': 2.0},
    'motion': {'motion_kernel_width': 9},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_transform(image, name):
    if name == 'dark':
        return ImageEnhance.Brightness(image).enhance(0.3)
    if name == 'low_contrast':
        return ImageEnhance.Contrast(image).enhance(0.35)
    if name == 'defocus':
        return image.filter(ImageFilter.GaussianBlur(2.0))
    if name == 'motion':
        array = np.asarray(image, dtype=np.float32)
        pad = np.pad(array, ((0, 0), (4, 4), (0, 0)), mode='edge')
        result = sum(pad[:, i:i + array.shape[1]] for i in range(9)) / 9
        return Image.fromarray(np.rint(result).astype(np.uint8))
    raise ValueError(f'Unknown transform: {name}')


def prepare():
    directory = ROOT / 'data/stress_tests'
    directory.mkdir(exist_ok=True)
    for condition in CONDITIONS:
        (directory / condition).mkdir(exist_ok=True)
        (directory / condition / '.gitkeep').touch()
    template = directory / 'manifest_template.csv'
    if not template.exists():
        with template.open('w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, FIELDS).writeheader()
    rows = []
    for label in CLASSES:
        for path in sorted((ROOT / 'data/test' / label).glob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'} or path.name.startswith('.'):
                continue
            rel = path.relative_to(ROOT).as_posix()
            object_id = f'trashnet:{label}:{path.stem}'
            rows.append(dict(path=rel, true_label=label, condition='standard', physical_object_id=object_id,
                             lighting='original', blur_level='original', capture_device='TrashNet',
                             notes='Original held-out split; object identity across original dataset not verified',
                             transform='none', source_path=rel, sha256=digest(path)))
            with Image.open(path) as src:
                image = src.convert('RGB')
            for transform in TRANSFORMS:
                target = directory / 'low_light_blur_synthetic' / f'{path.stem}_{transform}.png'
                apply_transform(image, transform).save(target)
                rows.append(dict(path=target.relative_to(ROOT).as_posix(), true_label=label,
                                 condition='low_light_blur_synthetic', physical_object_id=object_id,
                                 lighting='synthetic', blur_level=transform, capture_device='synthetic',
                                 notes=json.dumps(TRANSFORMS[transform]), transform=transform,
                                 source_path=rel, sha256=digest(target)))
    if not rows:
        raise ValueError('No held-out data/test images. Prepare dataset first.')
    # Human-labeled challenge manifests are maintained separately, never overwritten.
    for name in ['complex_manifest.csv', 'camera_manifest.csv', 'campus_manifest.csv']:
        path = directory / name
        if path.exists():
            with path.open(encoding='utf-8-sig', newline='') as f:
                rows.extend(csv.DictReader(f))
    with (directory / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader(); writer.writerows(rows)
    (directory / 'transform_parameters.json').write_text(json.dumps(TRANSFORMS, indent=2))
    return directory / 'manifest.csv'


def read_manifest(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'path', 'true_label', 'condition', 'physical_object_id'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest missing required columns.')
        rows = list(reader)
    seen, real_objects = set(), set()
    for row in rows:
        if row['true_label'] not in CLASSES or row['condition'] not in CONDITIONS:
            raise ValueError(f'Invalid ground truth or condition: {row}')
        p = (ROOT / row['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            raise ValueError(f'Missing or out-of-repository image: {row["path"]}')
        key = (row['condition'], str(p))
        if key in seen:
            raise ValueError(f'Duplicate image: {key}')
        seen.add(key)
        if row.get('sha256') and row['sha256'] != digest(p):
            raise ValueError(f'Image checksum changed: {p}')
        if row['condition'] in {'campus_novel', 'low_light_blur_real'}:
            obj = row['physical_object_id'].strip()
            if not obj or obj in real_objects:
                raise ValueError('Camera/campus sets require distinct, nonempty physical_object_id values.')
            if row.get('capture_device', '').lower() in {'', 'synthetic', 'trashnet'}:
                raise ValueError('Real camera rows require a capture device.')
            if any(p.is_relative_to(ROOT / 'data' / split) for split in ['train','validation','test','sample']):
                raise ValueError('Camera/campus images must not be from original dataset splits.')
            real_objects.add(obj)
    return rows


def metrics(rows):
    true = np.array([CLASSES.index(r['true_label']) for r in rows])
    pred = np.array([CLASSES.index(r['prediction']) for r in rows])
    return metrics_from_cm(confusion_matrix(true, pred))


def bootstrap(rows):
    # Cluster variants by source image, preserving dependence between transforms.
    groups = {}
    for r in rows:
        groups.setdefault(r['physical_object_id'], []).append(r)
    if len(groups) < 30 or set(r['true_label'] for r in rows) != set(CLASSES):
        return {'status': 'not reported: fewer than 30 source groups or incomplete class coverage'}
    values = list(groups.values())
    rng = np.random.default_rng(42)
    samples = [metrics([r for i in rng.integers(len(values), size=len(values)) for r in values[i]]) for _ in range(1000)]
    return {**{name: np.percentile([m[name] for m in samples], [2.5,97.5]).tolist() for name in ['accuracy','macro_f1']},
            'method': '1000 percentile source-image-cluster bootstrap resamples, seed 42; excludes seed/training uncertainty',
            'source_groups': len(groups)}


def save_plot(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_examples(rows, out, name, predicate, sort=False):
    selected = [r for r in rows if predicate(r)]
    if sort:
        selected.sort(key=lambda r:r['confidence'], reverse=True)
    selected = selected[:12]
    if not selected:
        return
    fig, axes = plt.subplots(3,4,figsize=(10,7))
    for ax in axes.flat:
        ax.axis('off')
    for ax,r in zip(axes.flat,selected):
        with Image.open(ROOT / r['path']) as img:
            ax.imshow(img.convert('RGB'))
        ax.set_title(f"{r['true_label']} -> {r['prediction']}\nscore {r['confidence']:.1%}",fontsize=8)
    save_plot(fig, out / f'{name}.png')


def evaluate(manifest, checkpoint, out, device):
    rows = read_manifest(manifest)
    if not rows:
        raise ValueError('Manifest has no labeled images; evaluation not performed.')
    out.mkdir(parents=True, exist_ok=True)
    # Remove only this evaluator's previous named figures, so missing scenarios
    # or newly error-free subsets cannot display stale galleries on a rerun.
    suffixes = ('confusion', 'per_class', 'confidence', 'correct', 'incorrect', 'high_confidence_errors')
    for condition in CONDITIONS:
        for suffix in suffixes:
            (out / f'{condition}_{suffix}.png').unlink(missing_ok=True)
    torch.set_num_threads(2)
    model = load_trained_model(checkpoint, device=device)
    with Image.open(ROOT / rows[0]['path']) as image:
        for _ in range(5):
            infer(model,image.convert('RGB'))
    predictions = []
    for row in rows:
        with Image.open(ROOT / row['path']) as image:
            # Dataset evaluation preserves original training conversion; upload EXIF/alpha
            # adjustments are tested independently, not silently applied to benchmark.
            result = infer(model,image.convert('RGB'))
        predictions.append({**row, 'prediction': result['label'], 'confidence':result['confidence'],
                            'seconds':result['seconds'], 'scores':result['scores'],
                            'image_sha256':digest(ROOT / row['path'])})
    summary = {'created_utc':datetime.now(timezone.utc).isoformat(), 'checkpoint':str(checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint),
               'checkpoint_sha256':digest(checkpoint), 'manifest_sha256':digest(manifest),
               'device':str(next(model.parameters()).device), 'torch':torch.__version__,
               'python':platform.python_version(), 'latency_protocol':'batch one; 5 warmups; synchronized forward + softmax; excludes decode, resize and host/device transfers',
               'scenarios':{}}
    comparison=[]
    for condition in CONDITIONS:
        subset = [r for r in predictions if r['condition']==condition]
        if not subset:
            summary['scenarios'][condition]={'n_samples':0,'status':'not evaluated; genuine labeled images required'}
            continue
        record = metrics(subset)
        record.update(mean_seconds=float(np.mean([r['seconds'] for r in subset])),
                      ci95=bootstrap(subset),confidence_quantiles=np.quantile([r['confidence'] for r in subset],[0,.25,.5,.75,1]).tolist())
        if condition == 'low_light_blur_synthetic':
            record['by_transform'] = {t:metrics([r for r in subset if r.get('transform')==t]) for t in TRANSFORMS if any(r.get('transform')==t for r in subset)}
        summary['scenarios'][condition] = record
        comparison.append({'scenario':condition,'n':len(subset),'accuracy':record['accuracy'],'macro_f1':record['macro_f1'],'mean_ms':record['mean_seconds']*1000})
        fig,ax=plt.subplots(figsize=(6,5)); cm=np.asarray(record['confusion_matrix'])
        ax.imshow(cm,cmap='Blues'); ax.set(xticks=range(6),yticks=range(6),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True',title=f'{condition} (n={len(subset)})'); plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
        for (i,j),value in np.ndenumerate(cm): ax.text(j,i,str(value),ha='center',va='center',color='white' if value>cm.max()/2 else 'black')
        save_plot(fig,out/f'{condition}_confusion.png')
        fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(6)
        for k,(field,title) in enumerate([('precision','Precision'),('recall','Recall'),('f1','F1')]):
            ax.bar(x+(k-1)*.25,[record['per_class'][c][field] for c in CLASSES],width=.25,label=title)
        ax.set(xticks=x,xticklabels=CLASSES,ylim=(0,1),title=condition); ax.legend(); save_plot(fig,out/f'{condition}_per_class.png')
        fig,ax=plt.subplots(figsize=(6,3))
        for correct in [True,False]:
            ax.hist([r['confidence'] for r in subset if (r['prediction']==r['true_label'])==correct], bins=np.linspace(0,1,21),alpha=.6,label='Correct' if correct else 'Incorrect')
        ax.set(xlabel='Uncalibrated maximum softmax score',ylabel='Count',title=condition); ax.legend(); save_plot(fig,out/f'{condition}_confidence.png')
        plot_examples(subset,out,f'{condition}_correct',lambda r:r['prediction']==r['true_label'])
        plot_examples(subset,out,f'{condition}_incorrect',lambda r:r['prediction']!=r['true_label'])
        plot_examples(subset,out,f'{condition}_high_confidence_errors',lambda r:r['prediction']!=r['true_label'],True)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
    with (out/'scenario_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,['scenario','n','accuracy','macro_f1','mean_ms']); w.writeheader(); w.writerows(comparison)
    for name,fields in [('scenario_comparison',['accuracy','macro_f1']),('latency',['mean_ms'])]:
        fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(comparison))
        for k,field in enumerate(fields): ax.bar(x+k*.3,[r[field] for r in comparison],width=.3,label=field)
        ax.set(xticks=x,xticklabels=[r['scenario'].replace('_','\n') for r in comparison],ylabel='Milliseconds' if name=='latency' else 'Score'); ax.legend(); save_plot(fig,out/f'{name}.png')
    print(json.dumps(comparison,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true',help='Build manifests and controlled variants; leaves original test files untouched')
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/stress_tests/manifest.csv')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/waste_cnn.pth')
    parser.add_argument('--out',type=Path,default=ROOT/'results/stress')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    args=parser.parse_args()
    if args.prepare: prepare()
    evaluate(args.manifest.resolve(),args.checkpoint.resolve(),args.out.resolve(),None if args.device=='auto' else args.device)
