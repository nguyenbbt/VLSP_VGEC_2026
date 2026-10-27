"""Self-contained S0 Modal images; importing does not start a GPU job."""
import os
from pathlib import Path
import modal
from .io import load_config

ARTIFACT_VOLUME = os.environ.get('VGEC_ARTIFACT_VOLUME', 'vgec-artifacts')
CACHE_VOLUME = os.environ.get('VGEC_CACHE_VOLUME', 'vgec-hf-cache')
GPU = os.environ.get('VGEC_GPU', 'L4')
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)


def project_root():
    return Path(__file__).resolve().parents[2]


def make_image(training=False):
    root = project_root()
    image = (modal.Image.debian_slim(python_version='3.11')
             .pip_install_from_requirements(str(root / 'requirements-modal-core.lock'))
             .env({'HF_HOME': '/hf-cache', 'PYTHONPATH': '/workspace:/workspace/src',
                   'TOKENIZERS_PARALLELISM': 'false', 'VGEC_GPU': GPU,
                   'VGEC_ARTIFACT_VOLUME': ARTIFACT_VOLUME, 'VGEC_CACHE_VOLUME': CACHE_VOLUME})
             .add_local_dir(root / 'src', remote_path='/workspace/src')
             .add_local_dir(root / 'configs', remote_path='/workspace/configs')
             .add_local_file(root / 'requirements-modal-core.lock', remote_path='/workspace/requirements-modal-core.lock')
             .add_local_file(root / 'modal_train.py', remote_path='/workspace/modal_train.py')
             .add_local_file(root / 'modal_predict.py', remote_path='/workspace/modal_predict.py'))
    if training:
        cfg = load_config(root / 'configs/data.yaml')
        names = [cfg['train'], cfg['dev'], cfg['prepared_dir'] + '/train.jsonl',
                 cfg['prepared_dir'] + '/dev.jsonl', cfg['report_dir'] + '/data_audit.json']
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Data paths must be relative to the S0 project')
            # A clean checkout can import/help without owning the training data.
            # CPU preflight must pass before submitting any training stage.
            if (root / relative).is_file():
                image = image.add_local_file(root / relative, remote_path='/workspace/' + relative.as_posix())
    return image
