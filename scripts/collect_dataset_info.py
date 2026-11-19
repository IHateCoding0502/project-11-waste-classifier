"""Count existing valid split images without modifying the dataset."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from starter.student_core import TrashNetDataset,CLASSES
if __name__=='__main__':
    result={}
    for split in ['train','validation','test']:
        if not (ROOT/'data'/split).exists():
            result[split]={'status':'missing'}
            continue
        data=TrashNetDataset(ROOT/'data'/split)
        result[split]={'n':len(data),'counts':{c:sum(y==i for _,y in data.samples) for i,c in enumerate(CLASSES)},'skipped':data.skipped}
    print(json.dumps(result,indent=2))
