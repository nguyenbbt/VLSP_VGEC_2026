import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class StandalonePackage(unittest.TestCase):
    def test_clean_checkout_imports_without_data_or_parent_project(self):
        with tempfile.TemporaryDirectory() as temp:
            isolated = Path(temp)
            shutil.copytree(ROOT/'src', isolated/'src', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(ROOT/'configs', isolated/'configs')
            for name in ('modal_train.py', 'modal_predict.py', 'requirements-modal-core.lock'):
                shutil.copy2(ROOT/name, isolated/name)
            env = dict(os.environ, PYTHONPATH=str(isolated/'src'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            code = ('from pathlib import Path; import modal_train, modal_predict; '
                    'assert modal_train.ROOT == Path.cwd(); '
                    'assert not Path("data").exists(); '
                    'from vgec.training.routing import MODELS; '
                    'assert MODELS == {"bartpho_attention", "vit5"}; '
                    'print("standalone OK")')
            result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], cwd=isolated,
                                    env=env, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('standalone OK', result.stdout)

    def test_inference_payload_excludes_answers_and_rejects_duplicate_ids(self):
        from modal_predict import source_payload
        row = {'ID': 5, 'Input': 'Example.', 'Output': {'private': 'answer'}}
        self.assertEqual(source_payload([row]), [{'ID': 5, 'Input': 'Example.'}])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            source_payload([row, row])
        with self.assertRaises(ValueError):
            source_payload([])

    def test_data_audit_preflight_and_stale_data_rejection(self):
        from vgec.data.audit import audit
        from vgec.training.routing import preflight
        from vgec.io import write_json
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT/'configs', root/'configs')
            def records(namespace, count):
                return [dict(ID=i, Input=f'source {namespace} {i}',
                             Output={'Error_Type_Id': i % 5, 'Fixed Sentence': [f'answer {namespace} {i}']})
                        for i in range(count)]
            write_json(root/'data/raw/final_data.json', records('train', 50))
            write_json(root/'data/raw/dev.json', records('dev', 10))
            previous = Path.cwd()
            try:
                os.chdir(root)
                audit('configs/data.yaml')
            finally:
                os.chdir(previous)
            report = preflight(root)
            self.assertEqual(report['prepared_train'], 50)
            self.assertEqual(report['splits']['bartpho_attention'], {'train': 40, 'validation': 10})
            with (root/'data/raw/final_data.json').open('a', encoding='utf-8') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'changed since audit'):
                preflight(root)

    def test_missing_data_fails_before_gpu(self):
        from vgec.training.routing import preflight
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT/'configs', root/'configs')
            with self.assertRaisesRegex(FileNotFoundError, 'run python -m vgec audit'):
                preflight(root)

    def test_workflow_only_smokes_and_trains_s0_models(self):
        from vgec.training.routing import run_sequence
        calls = []
        def invoke(model, run_id, steps, resume):
            calls.append((model, steps, resume))
            return dict(status='finished', smoke_only=bool(steps), optimizer_steps=steps or 100)
        self.assertEqual(len(run_sequence(invoke)), 4)
        self.assertEqual(calls, [('bartpho_attention', 20, True), ('vit5', 20, True),
                                 ('bartpho_attention', 0, True), ('vit5', 0, True)])

    def test_s0_prefers_bartpho_and_falls_back_to_copy_not_vit5(self):
        import torch
        from vgec.inference.pipeline import predict_with_models
        from vgec.inference.verifier import candidate
        from vgec.schemas import adapt
        row = adapt(dict(ID=1, Input='original'), 'test', supervised=False)
        model = SimpleNamespace(joint=True, eval=lambda: None,
                                predict_proba=lambda _: torch.tensor([[1., 0., 0., 0., 0.]]))
        models = [(model, None, {}, producer, 'cpu') for producer in
                  ('bartpho_corrector', 'vit5_corrector', 'bartpho_classifier')]
        for invalid in (False, True):
            def generate(net, tok, cfg, rows, device, producer, nbest):
                return [[candidate('original', producer, producer,
                                   truncated=invalid and producer == 'bartpho_corrector')]]
            with patch('vgec.inference.pipeline.encode', return_value={}), \
                 patch('vgec.inference.pipeline.generate_candidates', side_effect=generate):
                pred = predict_with_models([row], models, classifier_index=2, selector='s0')
            self.assertEqual(pred[0]['predicted_label'], 0)
            self.assertEqual(pred[0]['selected_text'], 'original' if invalid else 'bartpho_corrector')
            self.assertEqual(len(pred[0]['candidates']), 4)

    def test_modal_mounts_only_declared_training_inputs(self):
        from vgec import modal_runtime
        class Image:
            def __init__(self): self.mounts = []
            def pip_install_from_requirements(self, path):
                self.assert_no_mounts(); return self
            def env(self, values):
                self.assert_no_mounts(); return self
            def assert_no_mounts(self):
                if self.mounts: raise AssertionError('Build step follows a local mount')
            def add_local_file(self, path, remote_path):
                self.mounts.append((Path(path), remote_path)); return self
            def add_local_dir(self, path, remote_path):
                self.mounts.append((Path(path), remote_path)); return self
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT/'configs', root/'configs')
            for name in ('final_data.json', 'dev.json', 'private_test.json'):
                target = root/'data/raw'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('[]', encoding='utf-8')
            image = Image()
            with patch.object(modal_runtime, 'project_root', return_value=root), \
                 patch.object(modal_runtime.modal.Image, 'debian_slim', return_value=image):
                modal_runtime.make_image(training=True)
            mounts = {remote for _, remote in image.mounts}
            self.assertIn('/workspace/data/raw/final_data.json', mounts)
            self.assertNotIn('/workspace/data/raw/private_test.json', mounts)
            self.assertNotIn('/workspace/data', mounts)

    @unittest.skipUnless(shutil.which('git'), 'git not installed')
    def test_gitignore_excludes_data_weights_and_credentials_but_keeps_examples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(ROOT/'.gitignore', root/'.gitignore')
            subprocess.run(['git', 'init', '-q'], cwd=root, check=True, capture_output=True)
            ignored = ['data/raw/final_data.json', 'private_test.json', 'x/model.safetensors',
                       'x/optimizer.pt', '.env', '.modal.toml', 'reports/predictions.jsonl']
            for name in ignored:
                result = subprocess.run(['git', 'check-ignore', '-q', name], cwd=root)
                self.assertEqual(result.returncode, 0, name)
            for name in ['examples/input.json', 'src/vgec/cli.py', 'README.md']:
                result = subprocess.run(['git', 'check-ignore', '-q', name], cwd=root)
                self.assertEqual(result.returncode, 1, name)


if __name__ == '__main__':
    unittest.main()
