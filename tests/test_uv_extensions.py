"""Reject incomplete local Kit dependency graphs before launching the app."""
from pathlib import Path
import tempfile
import unittest

from verify_uv_runtime import verify_lab_extensions


class LocalExtensionTests(unittest.TestCase):
    def test_missing_transitive_extension_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'apps').mkdir()
            (root / 'apps/test.kit').write_text(
                '[dependencies]\n"isaaclab_tasks" = {}\n'
                '[settings]\na=1\n[settings]\nb=2\n')
            package = root / 'source/isaaclab_tasks'
            (package / 'config').mkdir(parents=True)
            (package / 'isaaclab_tasks').mkdir()
            (package / 'config/extension.toml').write_text(
                '[package]\nversion="1.0"\n[dependencies]\nisaaclab={}\n'
                '[[python.module]]\nname="isaaclab_tasks"\n')
            with self.assertRaisesRegex(RuntimeError, 'Missing local Kit extension isaaclab;'):
                verify_lab_extensions(root)
            core = root / 'source/isaaclab'
            (core / 'config').mkdir(parents=True)
            (core / 'isaaclab').mkdir()
            (core / 'config/extension.toml').write_text(
                '[package]\nversion="0.54.2"\n[[python.module]]\nname="isaaclab"\n')
            self.assertEqual(verify_lab_extensions(root), {'isaaclab_tasks': '1.0', 'isaaclab': '0.54.2'})
            (core / 'isaaclab').rmdir()
            with self.assertRaisesRegex(RuntimeError, 'Missing extension Python source'):
                verify_lab_extensions(root)
