"""赛事 Docker 推理入口与输出合同。"""

from rsdet.submission.competition import (
    CompetitionDetector,
    load_submission_config,
    run_submission,
    validate_result_payload,
)

__all__ = [
    "CompetitionDetector",
    "load_submission_config",
    "run_submission",
    "validate_result_payload",
]
