"""One atomic pointer document references immutable, checksummed snapshots."""
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from contextlib import contextmanager
from .io import write_json, read_json, file_hash, digest


@contextmanager
def local_lock(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    lock=directory/'active.lock'; owner=dict(pid=os.getpid(),owner=uuid.uuid4().hex,created=time.time())
    try:
        with lock.open('x',encoding='utf-8') as f:
            import json
            json.dump(owner,f)
    except FileExistsError:
        raise RuntimeError('Run is locked. Check owner before explicit stale-lock recovery: '+str(lock))
    try: yield owner
    finally:
        if lock.exists() and read_json(lock)==owner: lock.unlink()


def rng_state():
    import torch
    return dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import torch
    random.setstate(state['python']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def seed_all(seed):
    import torch
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class Store:
    def __init__(self,root,identity,commit=lambda:None):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.identity=identity; self.commit=commit
        self.pointer=self.root/'pointers.json'

    def resolve(self,kind='last'):
        if not self.pointer.exists(): return None
        pointers=read_json(self.pointer)
        if pointers['identity']!=self.identity: raise ValueError('Checkpoint experiment identity mismatch')
        name=pointers.get(kind)
        if not name: return None
        dest=(self.root/name).resolve()
        if dest.parent!=self.root.resolve(): raise ValueError('Invalid checkpoint pointer')
        manifest=read_json(dest/'complete.json')
        if manifest['identity']!=self.identity: raise ValueError('Snapshot identity mismatch')
        for file,expected in manifest['files'].items():
            p=(dest/file).resolve()
            if not p.is_relative_to(dest) or file_hash(p)!=expected: raise ValueError('Snapshot corruption: '+file)
        return dest

    def save(self,runtime,optimizer,scheduler,state,is_best=False):
        import torch
        start=time.monotonic(); name=f'step-{state["step"]:08d}-{uuid.uuid4().hex[:12]}'
        print(dict(stage='checkpoint_saving',job=self.root.name,step=state['step']),flush=True)
        tmp=self.root/('.tmp-'+name); tmp.mkdir()
        runtime.save_model(tmp)
        saved=dict(state=state,rng=rng_state(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())
        torch.save(saved,tmp/'training.pt')
        from .io import write_json
        write_json(tmp/'identity.json',dict(identity=self.identity,parameter_report=runtime.parameter_report))
        write_json(tmp/'progress.json',state)
        files={str(f.relative_to(tmp)).replace('\\','/'):file_hash(f) for f in tmp.rglob('*') if f.is_file()}
        write_json(tmp/'complete.json',dict(identity=self.identity,files=files))
        tmp.replace(self.root/name)
        self.commit()  # Snapshot reaches volume before pointer publication.
        old=read_json(self.pointer) if self.pointer.exists() else dict(identity=self.identity)
        new=dict(identity=self.identity,last=name,previous_last=old.get('last'),best=name if is_best else old.get('best'),last_save_seconds=time.monotonic()-start)
        write_json(self.pointer,new); self.commit()
        keep={new.get(k) for k in ('last','previous_last','best') if new.get(k)}
        for p in self.root.iterdir():
            if p.is_dir() and p.name.startswith('step-') and p.name not in keep:
                target=p.resolve()
                if target.parent!=self.root.resolve(): raise ValueError('Cleanup escaped checkpoint directory')
                shutil.rmtree(target)
        self.commit()
        seconds=time.monotonic()-start
        print(dict(stage='checkpoint_saved',job=self.root.name,step=state['step'],seconds=round(seconds,2)),flush=True)
        return seconds

    def load_training(self,destination,optimizer,scheduler):
        import torch
        data=torch.load(Path(destination)/'training.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(data['optimizer']); scheduler.load_state_dict(data['scheduler'])
        restore_rng(data['rng'])
        pointers=read_json(self.pointer)
        if pointers['last']==Path(destination).name and 'timing' in data['state']:
            data['state']['timing']['save']+=pointers.get('last_save_seconds',0.)
        return data['state']
