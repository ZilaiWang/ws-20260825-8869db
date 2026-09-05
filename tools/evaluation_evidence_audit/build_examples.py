"""Generate synthetic metadata examples only; no user dataset or model is read."""
from pathlib import Path
import copy
import hashlib
import json

root = Path(__file__).parent / 'examples'
root.mkdir(exist_ok=True)

def write(name, payload):
    (root / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

write('reference_output.json', {'records': [{'id': 'sample-a', 'value': 0.6}, {'id': 'sample-b', 'value': 0.4}]})
(root / 'reproduced_output.json').write_bytes((root / 'reference_output.json').read_bytes())
digest = hashlib.sha256((root / 'reference_output.json').read_bytes()).hexdigest()
safe = {
    'schema_version': 1,
    'notes': 'Synthetic metadata, not a result from the user project.',
    'datasets': [
        {'id': 'fit_set', 'groups': ['group-a', 'group-b'], 'role': 'training', 'disclosure': 'inspected'},
        {'id': 'selection_set', 'groups': ['group-c'], 'role': 'development', 'disclosure': 'inspected'},
        {'id': 'confirmation_set', 'groups': ['group-d'], 'role': 'confirmation', 'disclosure': 'untouched'},
    ],
    'nodes': [
        {'id': 'base_model', 'kind': 'fit', 'parents': [], 'exposure_datasets': ['fit_set'], 'lineage_complete': True},
        {'id': 'fixed_procedure', 'kind': 'select', 'parents': ['base_model'], 'exposure_datasets': ['selection_set'], 'lineage_complete': True},
    ],
    'claims': [
        {'id': 'generic_confirmation', 'candidate_node': 'fixed_procedure', 'evaluation_dataset': 'confirmation_set', 'claimed_role': 'independent_confirmation'},
    ],
    'artifacts': [
        {'id': 'reference', 'path': 'reference_output.json', 'expected_sha256': digest},
        {'id': 'reproduction', 'path': 'reproduced_output.json', 'expected_sha256': digest},
    ],
    'comparisons': [{'id': 'declared_output_parity', 'left': 'reference', 'right': 'reproduction', 'mode': 'json_exact'}],
    'signature_checks': [
        {'id': 'toy_training_comparison', 'required_fields': ['stage_epochs', 'initialization_kind'],
         'left': {'stage_epochs': [10, 5], 'initialization_kind': 'generic_pretrained'},
         'right': {'stage_epochs': [10, 5], 'initialization_kind': 'generic_pretrained'}},
    ],
    'test_coverage': [
        {'id': 'generic_control_flow', 'required_cases': ['ordinary', 'fallback', 'empty'], 'covered_cases': ['ordinary', 'fallback', 'empty']},
    ],
}
write('declared_clean.json', safe)
indirect = copy.deepcopy(safe)
indirect['nodes'] += [
    {'id': 'other_fold_model', 'kind': 'fit', 'parents': [], 'exposure_datasets': ['confirmation_set'], 'lineage_complete': True},
    {'id': 'selection_predictions', 'kind': 'predict', 'parents': ['other_fold_model'], 'exposure_datasets': ['selection_set'], 'lineage_complete': True},
]
indirect['nodes'][1]['parents'].append('selection_predictions')
write('indirect_overlap.json', indirect)
post_selection = copy.deepcopy(safe)
post_selection['nodes'].append({'id': 'posthoc_subset_choice', 'kind': 'select', 'parents': ['fixed_procedure'], 'exposure_datasets': ['confirmation_set'], 'lineage_complete': True})
post_selection['claims'][0]['candidate_node'] = 'posthoc_subset_choice'
post_selection['datasets'][2]['disclosure'] = 'inspected'
write('post_selection_reuse.json', post_selection)
unknown = copy.deepcopy(safe)
unknown['nodes'][0]['lineage_complete'] = False
unknown['signature_checks'][0]['right'].pop('stage_epochs')
unknown['test_coverage'][0]['covered_cases'] = ['ordinary']
write('unknown_and_missing_coverage.json', unknown)
