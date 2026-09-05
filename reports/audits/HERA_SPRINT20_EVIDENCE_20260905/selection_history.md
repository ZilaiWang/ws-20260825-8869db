# Sprint20 selection history

Date: 2026-09-05. This ledger preserves every material variant inspected in order. It is not a release decision.

|Order|Variant or question|Data inspected|Observed result|Decision at the time|Evidence role after audit|
|---:|---|---|---|---|---|
|1|Native OTO vs native OTM wiring|26 continuous images, 216 GT|Ship TP unchanged; OTM added 2 FP; Aircraft unchanged|Continue only as a mechanism probe|Integration sanity|
|2|Native OTM replaces all output|4481 full-seen images|Recall rose but FDR also rose|Do not replace the whole detector|Training-seen diagnostic|
|3|OTM owns all Ship classes 0–3|4481 full-seen|Score delta -0.1756|Reject broad Ship ownership|Training-seen selection input|
|4|OTM owns QHS/MS classes 2–3|4481 full-seen|Score delta +0.8811; +114 TP/+23 FP|Retain for OOF development|Training-seen selection input|
|5|OTM owns FSC class 24|4481 full-seen|Score delta +1.6423|Retain temporarily for OOF check|Training-seen selection input|
|6|OTM owns Ship and FSC|4481 full-seen|Score delta +1.4668|Do not prefer over narrower options|Training-seen selection input|
|7|All OTM policy|short three-fold OOF|Fold deltas -2.9933/+2.5309/+1.8621 at target 0.10|Reject as unstable|Development-selected OOF|
|8|OTM owns all Ship|short three-fold OOF|Fold deltas +0.4161/+0.6166/+1.9716|Reject after considering rare-class/full-seen risk|Development-selected OOF|
|9|OTM owns QHS/MS only|short three-fold OOF|Fold deltas +0.2985/+0.2155/+0.2625; group bootstrap positive probability 98.32%|Choose as narrow research candidate|Post-hoc development evidence, not independent confirmation|
|10|OTM owns FSC only|short three-fold OOF|Fold deltas -3.5072/+1.4532/-0.7300|Reject|Development-selected OOF|
|11|Exact fixed-primary QHS/MS policy at same Ship risk|short three-fold OOF, exact v7 replay|Fold deltas +0.2847/+0.1666/+0.2601; merged +0.2401; bootstrap positive probability 98.32%|Keep as mechanism evidence|Post-hoc development evidence|
|12|QHS/MS at arbitrary fixed macro-FDR targets|same short OOF|Deltas at 0.10/0.12/0.15/0.20 were -1.6044/-1.6660/-1.6691/-1.1726|Do not describe arbitrary-target results as positive|Development diagnostic|
|13|Shared OTO implementation|26 images then 4481 images|Exact parity on 4481/4481|Implementation parity passed|Implementation evidence only|
|14|Shared OTM implementation|26 images then 4481 images|61 images had coordinate differences|Reject shared-head deployment|Failed implementation parity|
|15|Bounded D4 early exit|26 easy images, historical CE hard100, current-consistency hard100|Exact outputs; matched hard100 local time +3.93%|Reject bounded D4 speed path|Engineering diagnostic|

The QHS/MS subset was chosen after several scopes and all three folds/full-seen diagnostics were inspected. Cross-fitted thresholds do not restore independence for the higher-level scope choice. The three fold fits also overlap in training sources, so their positive deltas are correlated, not three independent trials.
