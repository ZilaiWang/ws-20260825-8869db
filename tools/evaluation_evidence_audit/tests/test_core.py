from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from evidence_audit.core import (
    AuditInputError, audit_manifest, contained_file, json_differences, load_json, sha256_file,
)


def example() -> dict:
    return {
        'schema_version': 1,
        'datasets': [
            {'id': 'train', 'groups': ['A', 'B'], 'role': 'training', 'disclosure': 'inspected'},
            {'id': 'calibration', 'groups': ['C'], 'role': 'development', 'disclosure': 'inspected'},
            {'id': 'holdout', 'groups': ['D'], 'role': 'confirmation', 'disclosure': 'untouched'},
        ],
        'nodes': [
            {'id': 'model', 'kind': 'fit', 'parents': [], 'exposure_datasets': ['train'], 'lineage_complete': True},
            {'id': 'policy', 'kind': 'select', 'parents': ['model'], 'exposure_datasets': ['calibration'], 'lineage_complete': True},
        ],
        'claims': [
            {'id': 'final', 'candidate_node': 'policy', 'evaluation_dataset': 'holdout', 'claimed_role': 'independent_confirmation'},
        ],
    }


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_audit(self, data):
        return audit_manifest(data, self.root)

    def test_clean_metadata_is_not_certification(self):
        report = self.run_audit(example())
        self.assertEqual(report['errors'], 0)
        self.assertFalse(report['accuracy_or_deployment_approved'])
        self.assertEqual(report['claim_checks'][0]['status'], 'CONSISTENT_WITH_DECLARED_METADATA_ONLY')

    def test_training_exposure(self):
        data = example()
        data['datasets'][0]['groups'].append('D')
        report = self.run_audit(data)
        self.assertEqual(report['claim_checks'][0]['status'], 'CONTRADICTED')
        self.assertEqual(report['errors'], 1)

    def test_selection_exposure(self):
        data = example()
        data['nodes'][1]['exposure_datasets'].append('holdout')
        self.assertEqual(self.run_audit(data)['claim_checks'][0]['status'], 'CONTRADICTED')

    def test_indirect_exposure_from_calibration_predictor(self):
        data = example()
        data['nodes'] += [
            {'id': 'other_fold_model', 'kind': 'fit', 'parents': [], 'exposure_datasets': ['holdout'], 'lineage_complete': True},
            {'id': 'calibration_predictions', 'kind': 'predict', 'parents': ['other_fold_model'], 'exposure_datasets': ['calibration'], 'lineage_complete': True},
        ]
        data['nodes'][1]['parents'].append('calibration_predictions')
        report = self.run_audit(data)
        found = report['claim_checks'][0]['overlapping_exposures']
        self.assertTrue(any(x['dependency_path'] == ['policy', 'calibration_predictions', 'other_fold_model'] for x in found))

    def test_inspected_holdout(self):
        data = example()
        data['datasets'][2]['disclosure'] = 'inspected'
        self.assertEqual(self.run_audit(data)['claim_checks'][0]['status'], 'NOT_AN_UNTOUCHED_CONFIRMATION_SET')

    def test_unknown_disclosure(self):
        data = example()
        data['datasets'][2]['disclosure'] = 'unknown'
        self.assertEqual(self.run_audit(data)['claim_checks'][0]['status'], 'UNKNOWN')

    def test_incomplete_ancestry(self):
        data = example()
        data['nodes'][0]['lineage_complete'] = False
        self.assertEqual(self.run_audit(data)['claim_checks'][0]['status'], 'UNKNOWN')

    def test_development_claim_not_certified(self):
        data = example()
        data['datasets'][2]['disclosure'] = 'inspected'
        data['claims'][0]['claimed_role'] = 'development'
        report = self.run_audit(data)
        self.assertEqual(report['errors'], 0)
        self.assertGreater(report['warnings'], 0)

    def test_training_role_is_not_confirmation(self):
        data = example()
        data['datasets'][2]['role'] = 'training'
        self.assertEqual(self.run_audit(data)['claim_checks'][0]['status'], 'NOT_AN_UNTOUCHED_CONFIRMATION_SET')

    def test_cycle_rejected(self):
        data = example()
        data['nodes'][0]['parents'].append('policy')
        with self.assertRaises(AuditInputError):
            self.run_audit(data)

    def test_cycle_in_unused_node_rejected(self):
        data = example()
        data['nodes'].append({'id': 'unused', 'kind': 'fit', 'parents': ['unused'], 'exposure_datasets': [], 'lineage_complete': True})
        with self.assertRaises(AuditInputError):
            self.run_audit(data)

    def test_duplicate_node_rejected(self):
        data = example()
        data['nodes'].append(copy.deepcopy(data['nodes'][0]))
        with self.assertRaises(AuditInputError):
            self.run_audit(data)

    def test_unknown_parent_rejected(self):
        data = example()
        data['nodes'][1]['parents'].append('missing')
        with self.assertRaises(AuditInputError):
            self.run_audit(data)

    def test_duplicate_group_rejected(self):
        data = example()
        data['datasets'][0]['groups'].append('A')
        with self.assertRaises(AuditInputError):
            self.run_audit(data)

    def test_empty_groups_rejected(self):
        data = example()
        data['datasets'][0]['groups'] = []
        with self.assertRaises(AuditInputError):
            self.run_audit(data)

    def test_bool_schema_rejected(self):
        data = example()
        data['schema_version'] = True
        with self.assertRaises(AuditInputError):
            self.run_audit(data)

    def test_bool_lineage_required(self):
        data = example()
        data['nodes'][0]['lineage_complete'] = 'true'
        with self.assertRaises(AuditInputError):
            self.run_audit(data)

    def test_nonfinite_in_memory_rejected(self):
        data = example()
        data['extra'] = float('nan')
        with self.assertRaises(AuditInputError):
            self.run_audit(data)

    def test_nonfinite_json_rejected(self):
        file = self.root / 'bad.json'
        file.write_text('{"x": NaN}')
        with self.assertRaises(AuditInputError):
            load_json(file)

    def test_overflow_json_rejected(self):
        file = self.root / 'bad.json'
        file.write_text('{"x": 1e9999}')
        with self.assertRaises(AuditInputError):
            load_json(file)

    def test_duplicate_json_key_rejected(self):
        file = self.root / 'bad.json'
        file.write_text('{"x": 1, "x": 2}')
        with self.assertRaises(AuditInputError):
            load_json(file)

    def test_path_escape_rejected(self):
        with self.assertRaises(AuditInputError):
            contained_file(self.root, '../other.txt')

    def test_absolute_path_rejected(self):
        with self.assertRaises(AuditInputError):
            contained_file(self.root, '/etc/passwd')

    def test_symlink_escape_rejected(self):
        (self.root / 'escape').symlink_to('/etc/passwd')
        with self.assertRaises(AuditInputError):
            contained_file(self.root, 'escape')

    def test_file_not_exists(self):
        with self.assertRaises(AuditInputError):
            contained_file(self.root, 'missing')

    def test_sha_and_byte_comparison(self):
        (self.root / 'left').write_bytes(b'abc')
        (self.root / 'right').write_bytes(b'abc')
        data = example()
        digest = sha256_file(self.root / 'left')
        data['artifacts'] = [
            {'id': 'left', 'path': 'left', 'expected_sha256': digest},
            {'id': 'right', 'path': 'right', 'expected_sha256': digest},
        ]
        data['comparisons'] = [{'id': 'eq', 'left': 'left', 'right': 'right', 'mode': 'bytes'}]
        self.assertTrue(self.run_audit(data)['comparisons'][0]['equal'])

    def test_hash_mismatch_reported(self):
        (self.root / 'a').write_bytes(b'a')
        data = example()
        data['artifacts'] = [{'id': 'a', 'path': 'a', 'expected_sha256': '0' * 64}]
        self.assertEqual(self.run_audit(data)['artifact_checks']['a']['status'], 'HASH_MISMATCH')

    def test_missing_previous_hash_is_not_attestation(self):
        (self.root / 'a').write_bytes(b'a')
        data = example()
        data['artifacts'] = [{'id': 'a', 'path': 'a'}]
        report = self.run_audit(data)
        self.assertEqual(report['artifact_checks']['a']['status'], 'NO_PREVIOUS_HASH_ATTESTATION')
        self.assertEqual(report['warnings'], 1)

    def test_json_formatting_not_byte_equal(self):
        (self.root / 'left').write_text('{"x":1,"y":2}')
        (self.root / 'right').write_text('{"y": 2, "x": 1}\n')
        data = example()
        data['artifacts'] = [{'id': 'left', 'path': 'left'}, {'id': 'right', 'path': 'right'}]
        data['comparisons'] = [{'id': 'eq', 'left': 'left', 'right': 'right', 'mode': 'json_exact'}]
        result = self.run_audit(data)['comparisons'][0]
        self.assertTrue(result['equal'])
        self.assertFalse(result['byte_equal'])

    def test_array_order_not_ignored(self):
        self.assertTrue(json_differences([1, 2], [2, 1]))

    def test_array_multiplicity_not_ignored(self):
        self.assertTrue(json_differences([1, 1], [1]))

    def test_bool_is_not_integer(self):
        self.assertTrue(json_differences(True, 1))

    def test_missing_is_not_null(self):
        diff = json_differences({}, {'a': None})
        self.assertTrue(diff[0]['left_missing'])

    def test_json_pointer_escape(self):
        self.assertEqual(json_differences({'a/b~c': 1}, {'a/b~c': 2})[0]['path'], '/a~1b~0c')

    def test_missing_signature_not_equal(self):
        data = example()
        data['signature_checks'] = [{'id': 's', 'required_fields': ['epochs'], 'left': {}, 'right': {}}]
        self.assertEqual(self.run_audit(data)['signature_checks'][0]['status'], 'UNKNOWN')

    def test_different_training_signatures(self):
        data = example()
        data['signature_checks'] = [{'id': 's', 'required_fields': ['epochs'], 'left': {'epochs': 12}, 'right': {'epochs': 40}}]
        self.assertEqual(self.run_audit(data)['signature_checks'][0]['status'], 'DIFFERENT')

    def test_equal_signatures(self):
        data = example()
        data['signature_checks'] = [{'id': 's', 'required_fields': ['stages'], 'left': {'stages': [1, 2]}, 'right': {'stages': [1, 2]}}]
        self.assertEqual(self.run_audit(data)['signature_checks'][0]['status'], 'DECLARED_FIELDS_EQUAL')

    def test_missing_test_branch_reported(self):
        data = example()
        data['test_coverage'] = [{'id': 'cases', 'required_cases': ['ordinary', 'fallback'], 'covered_cases': ['ordinary']}]
        self.assertEqual(self.run_audit(data)['test_coverage'][0]['missing_cases'], ['fallback'])

    def test_empty_manifest_does_not_certify(self):
        self.assertFalse(self.run_audit({'schema_version': 1})['accuracy_or_deployment_approved'])

    def test_unknown_artifact_comparison_rejected(self):
        data = example()
        data['comparisons'] = [{'id': 'bad', 'left': 'a', 'right': 'b', 'mode': 'bytes'}]
        with self.assertRaises(AuditInputError):
            self.run_audit(data)


if __name__ == '__main__':
    unittest.main()
