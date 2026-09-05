from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'input.json'
        self.manifest.write_text(json.dumps({'schema_version': 1}))
        self.out = self.root / 'report.json'

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *extra):
        return subprocess.run(
            [sys.executable, '-m', 'evidence_audit', '--manifest', str(self.manifest),
             '--root', str(self.root), '--out', str(self.out), *extra],
            cwd=ROOT, text=True, capture_output=True, timeout=10,
        )

    def test_output_created_and_never_certifies(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(self.out.read_text())['accuracy_or_deployment_approved'])

    def test_output_reuse_rejected(self):
        self.out.write_text('preserve me')
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertEqual(self.out.read_text(), 'preserve me')

    def test_fail_on_warning(self):
        self.assertEqual(self.run_cli('--fail-on-warning').returncode, 2)

    def test_invalid_input_no_output(self):
        self.manifest.write_text('{"schema_version":1,"schema_version":1}')
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertFalse(self.out.exists())

    def test_manifest_not_overwritten(self):
        before = self.manifest.read_bytes()
        self.out = self.manifest
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_unknown_field_rejected(self):
        self.manifest.write_text('{"schema_version":1,"claim":[]}')
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
