"""Verify selected checkpoint shape, parameters and CPU/CUDA inference."""
import argparse
import json
import sys
from pathlib import Path
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from starter.student_core import load_trained_model, count_parameters, CLASSES
from starter.inference import infer

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/waste_cnn.pth')
    parser.add_argument('--image',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        model=load_trained_model(args.checkpoint,device=device)
        with torch.inference_mode():
            logits=model(torch.zeros(1,3,*model.image_size,device=device))
        assert logits.shape==(1,6) and torch.isfinite(logits).all()
        image=Image.open(args.image).convert('RGB') if args.image else Image.new('RGB',(64,64))
        result=infer(model,image)
        assert list(result['scores'])==list(CLASSES)
        print(json.dumps({'device':device,'params':count_parameters(model),'shape':list(logits.shape),'result':result}))
