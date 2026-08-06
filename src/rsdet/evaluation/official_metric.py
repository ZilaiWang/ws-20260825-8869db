"""比赛 Recall/FDR 评估。

官方规则要求先在 25 个细类内完成匹配，再按 ship、aircraft、vehicle 三大类
汇总 TP、FP、FN。预测按分数降序贪心匹配；预测与 GT 的细类 ``category_id``
必须相同；每个 GT 只匹配一次；重复框计为 FP。三大类映射只用于选择 IoU
阈值和汇总指标，不能在匹配前消除细类。

评分方案 V1.6 进一步明确了官方排名口径：三大类各自的 Recall 与 FDR =
**大类内细类指标的简单平均**（船 4 型各 1/4、飞机 20 型各 1/20、车辆 1 型即
FSC 本身），用于 7 排名二次排序；刚性门槛（Recall≥0.85 / FDR≤0.20）仍按
三类合并 pooled 计算。因此本模块同时提供 pooled（``evaluate_predictions``）
与官方排名（``evaluate_ranking_metrics``）两种聚合，二者共用同一匹配轨迹。
"""

import math
from dataclasses import dataclass, field
from typing import Any

IOU_THRESHOLDS = {
    "ship": 0.50,
    "aircraft": 0.50,
    "vehicle": 0.35,
}


@dataclass
class PerClassMetrics:
    """单类 TP、FP、FN 及其派生指标。"""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def recall(self) -> float:
        """返回 Recall；没有 GT 时按 1.0 处理。"""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 1.0

    @property
    def fdr(self) -> float:
        """返回 FDR；没有预测时按 0.0 处理。"""
        denominator = self.fp + self.tp
        return self.fp / denominator if denominator else 0.0


@dataclass
class OverallMetrics:
    """总体和分大类评估结果。"""

    recall: float = 0.0
    fdr: float = 0.0
    per_class: dict[str, PerClassMetrics] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OfficialMatch:
    """一次官方贪心匹配及其原始列表位置。

    ``prediction_index`` 和 ``ground_truth_index`` 都是对应 ``image_id``
    下输入列表中的零基下标。它们让诊断模块能够复用同一套官方匹配，而
    不必重新实现或猜测匹配归属。
    """

    image_id: int
    class_name: str
    category_id: int
    prediction_index: int
    ground_truth_index: int
    score: float
    iou: float


@dataclass(frozen=True)
class OfficialUnmatchedPrediction:
    """官方匹配后仍未命中的一条预测。"""

    image_id: int
    class_name: str
    category_id: int
    prediction_index: int
    score: float


@dataclass(frozen=True)
class OfficialUnmatchedGroundTruth:
    """官方匹配后仍未命中的一条 GT。"""

    image_id: int
    class_name: str
    category_id: int
    ground_truth_index: int


@dataclass(frozen=True)
class OfficialEvaluationTrace:
    """与 ``OverallMetrics`` 同源的逐对象官方匹配轨迹。"""

    matches: tuple[OfficialMatch, ...]
    unmatched_predictions: tuple[OfficialUnmatchedPrediction, ...]
    unmatched_ground_truths: tuple[OfficialUnmatchedGroundTruth, ...]


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """计算两个 xyxy 像素框的 IoU。"""
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("bbox 必须包含 4 个数值")

    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter_area = max(0.0, xb - xa) * max(0.0, yb - ya)

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """兼容旧内部调用；新诊断代码应使用公开 ``compute_iou``。"""

    return compute_iou(box_a, box_b)


def evaluate_predictions(
    gt_boxes: dict[int, list[dict[str, Any]]],
    pred_boxes: dict[int, list[dict[str, Any]]],
    class_names: list[str] | None = None,
    category_mapping: dict[int, str] | None = None,
    iou_thresholds: dict[str, float] | None = None,
) -> OverallMetrics:
    """按官方细类匹配规则计算并汇总比赛指标。

    Args:
        gt_boxes: ``{image_id: [{bbox_xyxy, category_id}, ...]}``。
        pred_boxes: ``{image_id: [{bbox_xyxy, score, category_id}, ...]}``。
        class_names: 参与评估的大类，默认 ship、aircraft、vehicle。
        category_mapping: 数据集 category_id 到大类名称的映射。省略时仅接受
            ``0=ship, 1=aircraft, 2=vehicle`` 的三类输出。
        iou_thresholds: 三大类 IoU 阈值。省略时使用官方默认值：舰船/飞机
            0.50，车辆 0.35。

    Raises:
        ValueError: 类别映射缺失或包含未知大类。
    """
    result, _ = evaluate_predictions_with_trace(
        gt_boxes,
        pred_boxes,
        class_names=class_names,
        category_mapping=category_mapping,
        iou_thresholds=iou_thresholds,
    )
    return result


def evaluate_predictions_with_trace(
    gt_boxes: dict[int, list[dict[str, Any]]],
    pred_boxes: dict[int, list[dict[str, Any]]],
    class_names: list[str] | None = None,
    category_mapping: dict[int, str] | None = None,
    iou_thresholds: dict[str, float] | None = None,
) -> tuple[OverallMetrics, OfficialEvaluationTrace]:
    """计算官方指标并返回完全同源的逐对象匹配轨迹。

    该函数与 :func:`evaluate_predictions` 使用相同的细类约束、分数降序、
    一对一贪心和 IoU 阈值。轨迹仅增加可审计性，不改变官方指标。
    """

    names = class_names or list(IOU_THRESHOLDS)
    unknown_names = set(names) - set(IOU_THRESHOLDS)
    if unknown_names:
        raise ValueError(f"未知评估类别: {sorted(unknown_names)}")

    thresholds = dict(IOU_THRESHOLDS if iou_thresholds is None else iou_thresholds)
    missing_thresholds = set(names) - set(thresholds)
    if missing_thresholds:
        raise ValueError(f"评估类别缺少 IoU 阈值: {sorted(missing_thresholds)}")
    for name in names:
        threshold = float(thresholds[name])
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"非法 IoU 阈值 {name}={threshold}")
        thresholds[name] = threshold

    mapping = (
        {index: name for index, name in enumerate(names)}
        if category_mapping is None
        else category_mapping
    )
    if not mapping:
        raise ValueError("category_mapping 不能为空")
    normalized_mapping = {int(category_id): name for category_id, name in mapping.items()}
    unknown_targets = set(normalized_mapping.values()) - set(names)
    if unknown_targets:
        raise ValueError(f"类别映射包含未参与评估的大类: {sorted(unknown_targets)}")

    normalized_gt = _normalize_records(gt_boxes, normalized_mapping, require_score=False)
    normalized_pred = _normalize_records(pred_boxes, normalized_mapping, require_score=True)

    per_class: dict[str, PerClassMetrics] = {}
    all_matches: list[OfficialMatch] = []
    all_unmatched_predictions: list[OfficialUnmatchedPrediction] = []
    all_unmatched_ground_truths: list[OfficialUnmatchedGroundTruth] = []
    total_tp = total_fp = total_fn = 0
    for class_name in names:
        (
            matches,
            unmatched_predictions,
            unmatched_ground_truths,
        ) = _evaluate_per_class_with_trace(
            normalized_gt,
            normalized_pred,
            class_name,
            thresholds[class_name],
        )
        tp = len(matches)
        fp = len(unmatched_predictions)
        fn = len(unmatched_ground_truths)
        per_class[class_name] = PerClassMetrics(tp=tp, fp=fp, fn=fn)
        all_matches.extend(matches)
        all_unmatched_predictions.extend(unmatched_predictions)
        all_unmatched_ground_truths.extend(unmatched_ground_truths)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    recall_denominator = total_tp + total_fn
    fdr_denominator = total_tp + total_fp
    result = OverallMetrics(
        recall=total_tp / recall_denominator if recall_denominator else 1.0,
        fdr=total_fp / fdr_denominator if fdr_denominator else 0.0,
        per_class=per_class,
        details={
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "total_gt": recall_denominator,
            "total_pred": fdr_denominator,
            "matching_policy": "same_fine_category_id",
            "aggregation_policy": "fine_match_then_coarse_and_overall",
            "iou_thresholds": {name: thresholds[name] for name in names},
            "empty_gt_recall_policy": 1.0,
            "empty_prediction_fdr_policy": 0.0,
        },
    )
    trace = OfficialEvaluationTrace(
        matches=tuple(all_matches),
        unmatched_predictions=tuple(all_unmatched_predictions),
        unmatched_ground_truths=tuple(all_unmatched_ground_truths),
    )
    return result, trace


def _normalize_records(
    records: dict[int, list[dict[str, Any]]],
    category_mapping: dict[int, str],
    *,
    require_score: bool,
) -> dict[int, list[dict[str, Any]]]:
    """校验记录并把细类 ID 归并为三大类名称。"""
    normalized: dict[int, list[dict[str, Any]]] = {}
    for image_id, items in records.items():
        normalized[image_id] = []
        for source_index, item in enumerate(items):
            category_id = int(item["category_id"])
            if category_id not in category_mapping:
                raise ValueError(f"category_id={category_id} 缺少三大类映射")
            box = [float(value) for value in item["bbox_xyxy"]]
            if (
                len(box) != 4
                or not all(math.isfinite(value) for value in box)
                or box[2] < box[0]
                or box[3] < box[1]
            ):
                raise ValueError(f"非法 xyxy bbox: {box}")
            normalized_item: dict[str, Any] = {
                "bbox_xyxy": box,
                "category_id": category_id,
                "class_name": category_mapping[category_id],
                "source_index": source_index,
            }
            if require_score:
                score = float(item["score"])
                if not math.isfinite(score):
                    raise ValueError(f"非法 score: {score}")
                normalized_item["score"] = score
            normalized[image_id].append(normalized_item)
    return normalized


def _evaluate_per_class(
    gt_boxes: dict[int, list[dict[str, Any]]],
    pred_boxes: dict[int, list[dict[str, Any]]],
    class_name: str,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """在同一细类内匹配，并为一个大类累计 TP、FP、FN。"""
    matches, unmatched_predictions, unmatched_ground_truths = (
        _evaluate_per_class_with_trace(
            gt_boxes,
            pred_boxes,
            class_name,
            iou_threshold,
        )
    )
    return len(matches), len(unmatched_predictions), len(unmatched_ground_truths)


def _evaluate_per_class_with_trace(
    gt_boxes: dict[int, list[dict[str, Any]]],
    pred_boxes: dict[int, list[dict[str, Any]]],
    class_name: str,
    iou_threshold: float,
) -> tuple[
    list[OfficialMatch],
    list[OfficialUnmatchedPrediction],
    list[OfficialUnmatchedGroundTruth],
]:
    """执行单个大类的官方匹配并保留输入列表位置。"""

    matches: list[OfficialMatch] = []
    unmatched_predictions: list[OfficialUnmatchedPrediction] = []
    unmatched_ground_truths: list[OfficialUnmatchedGroundTruth] = []
    for image_id in sorted(set(gt_boxes) | set(pred_boxes)):
        gts = [
            item
            for item in gt_boxes.get(image_id, [])
            if item["class_name"] == class_name
        ]
        predictions = [
            item for item in pred_boxes.get(image_id, []) if item["class_name"] == class_name
        ]
        predictions.sort(key=lambda item: item["score"], reverse=True)
        matched = [False] * len(gts)

        for prediction in predictions:
            best_index = -1
            best_iou = -1.0
            for index, gt in enumerate(gts):
                if matched[index]:
                    continue
                if prediction["category_id"] != gt["category_id"]:
                    continue
                iou = _compute_iou(prediction["bbox_xyxy"], gt["bbox_xyxy"])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index >= 0:
                matched[best_index] = True
                gt = gts[best_index]
                matches.append(
                    OfficialMatch(
                        image_id=image_id,
                        class_name=class_name,
                        category_id=int(prediction["category_id"]),
                        prediction_index=int(prediction["source_index"]),
                        ground_truth_index=int(gt["source_index"]),
                        score=float(prediction["score"]),
                        iou=float(best_iou),
                    )
                )
            else:
                unmatched_predictions.append(
                    OfficialUnmatchedPrediction(
                        image_id=image_id,
                        class_name=class_name,
                        category_id=int(prediction["category_id"]),
                        prediction_index=int(prediction["source_index"]),
                        score=float(prediction["score"]),
                    )
                )

        for index, is_matched in enumerate(matched):
            if is_matched:
                continue
            gt = gts[index]
            unmatched_ground_truths.append(
                OfficialUnmatchedGroundTruth(
                    image_id=image_id,
                    class_name=class_name,
                    category_id=int(gt["category_id"]),
                    ground_truth_index=int(gt["source_index"]),
                )
            )

    return matches, unmatched_predictions, unmatched_ground_truths


@dataclass
class FineClassMetrics:
    """单个细类的 TP、FP、FN 及其派生指标。

    官方排名口径的最小单位：大类指标 = 大类内各细类指标的简单平均。
    空 GT 细类按 Recall=1.0 / FDR=0.0 处理，与 :class:`PerClassMetrics`
    的空策略一致；不过 :func:`evaluate_ranking_metrics` 只把 GT 中
    出现过的细类纳入平均，0-GT 细类不参与（见 ``details``）。
    """

    category_id: int
    coarse_class: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 1.0

    @property
    def fdr(self) -> float:
        denominator = self.fp + self.tp
        return self.fp / denominator if denominator else 0.0


@dataclass
class CoarseMacroMetrics:
    """一个大类的官方排名口径汇总（macro）及 pooled 对照。"""

    macro_recall: float
    macro_fdr: float
    pooled_recall: float
    pooled_fdr: float
    fine_count: int
    fine_ids: list[int]


@dataclass
class RankingMetrics:
    """官方 V1.6 排名口径评估结果。

    - ``per_fine``：细类级 TP/FP/FN 与派生指标。
    - ``per_coarse``：各大类的 macro（细类简单平均）与 pooled 对照。
    - ``overall_recall`` / ``overall_fdr``：全部参与细类的简单平均，是
      团队自定义的"官方口径 Overall"，用于内部目标（如 FDR≤0.17）追踪；
      官方排名本身只定义到大类级（7 项排名）。
    """

    per_fine: dict[int, FineClassMetrics] = field(default_factory=dict)
    per_coarse: dict[str, CoarseMacroMetrics] = field(default_factory=dict)
    overall_recall: float = 0.0
    overall_fdr: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_ranking_metrics(
    gt_boxes: dict[int, list[dict[str, Any]]],
    pred_boxes: dict[int, list[dict[str, Any]]],
    class_names: list[str] | None = None,
    category_mapping: dict[int, str] | None = None,
    iou_thresholds: dict[str, float] | None = None,
) -> RankingMetrics:
    """按官方评分方案 V1.6 排名口径计算指标。

    匹配规则与 :func:`evaluate_predictions` 完全一致（同源 trace），差异只在
    聚合层：细类级 TP/FP/FN 先按细类算出 Recall/FDR，再在大类内简单平均。
    参与平均的细类 = GT 中出现过的细类（0-GT 细类不参与，避免空类稀释）。

    Args 与 :func:`evaluate_predictions` 相同。
    """
    result, trace = evaluate_predictions_with_trace(
        gt_boxes,
        pred_boxes,
        class_names=class_names,
        category_mapping=category_mapping,
        iou_thresholds=iou_thresholds,
    )

    names = result.per_class.keys()

    # GT 中出现过的细类才参与 macro 平均。
    present_fine: set[int] = set()
    for items in gt_boxes.values():
        for item in items:
            present_fine.add(int(item["category_id"]))

    per_fine: dict[int, FineClassMetrics] = {}
    for match in trace.matches:
        key = int(match.category_id)
        metrics = per_fine.setdefault(
            key, FineClassMetrics(key, match.class_name)
        )
        metrics.tp += 1
    for prediction in trace.unmatched_predictions:
        key = int(prediction.category_id)
        metrics = per_fine.setdefault(
            key, FineClassMetrics(key, prediction.class_name)
        )
        metrics.fp += 1
    for ground_truth in trace.unmatched_ground_truths:
        key = int(ground_truth.category_id)
        metrics = per_fine.setdefault(
            key, FineClassMetrics(key, ground_truth.class_name)
        )
        metrics.fn += 1

    coarse_fine_ids: dict[str, list[int]] = {}
    for key, metrics in per_fine.items():
        coarse_fine_ids.setdefault(metrics.coarse_class, []).append(key)

    per_coarse: dict[str, CoarseMacroMetrics] = {}
    overall_recall_sum = overall_fdr_sum = 0.0
    overall_fine_count = 0
    for coarse_name in names:
        fine_ids = sorted(coarse_fine_ids.get(coarse_name, []))
        if not fine_ids:
            continue
        participating = [fine_id for fine_id in fine_ids if fine_id in present_fine]
        if not participating:
            continue
        macro_recall = sum(per_fine[fine_id].recall for fine_id in participating) / len(
            participating
        )
        macro_fdr = sum(per_fine[fine_id].fdr for fine_id in participating) / len(
            participating
        )
        pooled = result.per_class[coarse_name]
        per_coarse[coarse_name] = CoarseMacroMetrics(
            macro_recall=macro_recall,
            macro_fdr=macro_fdr,
            pooled_recall=pooled.recall,
            pooled_fdr=pooled.fdr,
            fine_count=len(participating),
            fine_ids=participating,
        )
        overall_recall_sum += macro_recall * len(participating)
        overall_fdr_sum += macro_fdr * len(participating)
        overall_fine_count += len(participating)

    return RankingMetrics(
        per_fine=per_fine,
        per_coarse=per_coarse,
        overall_recall=(
            overall_recall_sum / overall_fine_count if overall_fine_count else 1.0
        ),
        overall_fdr=(
            overall_fdr_sum / overall_fine_count if overall_fine_count else 0.0
        ),
        details={
            "matching_policy": result.details["matching_policy"],
            "aggregation_policy": "official_ranking_v1_6_fine_macro_average",
            "fine_average_policy": "present_in_gt_only",
            "empty_gt_recall_policy": result.details["empty_gt_recall_policy"],
            "empty_prediction_fdr_policy": result.details[
                "empty_prediction_fdr_policy"
            ],
            "iou_thresholds": result.details["iou_thresholds"],
        },
    )
