import os,sys,json
sys.path.append(os.getcwd())
from os.path import join,exists
import pathlib
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import re, math
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')
from ipdb import set_trace

from torch.utils.data import DataLoader
import transformers

from configs.unified_config import ModelArguments,DataArguments,TrainingArguments,InferenceArguments

from dataset.unified_dataset import get_dataset_collator
from utils.util import set_seed,find_all_linear_names,prepare_sample,write2json,load_ckpt
from utils.avss_utils import (
    mask_iou,compute_miou_from_jsonl,calc_color_miou_fscore,
    save_color_mask,save_gt_mask,Eval_Fmeasure,
    metric_s_for_null
)
from utils.deepspeed_utils import *

local_rank = None


def _normalize_action(action_str: str):
    if action_str is None:
        return None
    action_str = action_str.lower()
    if 'accept' in action_str:
        return 'accept'
    if 'major' in action_str:
        return 'major revision'
    if 'minor' in action_str:
        return 'minor revision'
    if 'reject' in action_str:
        return 'reject'
    return action_str.strip()


def _normalize_mask_type(mask_type: str):
    if mask_type is None:
        return None
    m = mask_type.lower()
    if 'perfect' in m:
        return 'perfect'
    if 'cut' in m:
        return 'cutoff' if 'cutoff' in m else 'cutout'
    if 'erode' in m:
        return 'erode'
    if 'dilate' in m:
        return 'dilate'
    if 'merge' in m:
        return 'merge'
    if m in ['null', 'null_mask', 'empty', 'empty_mask']:
        return 'null'
    if 'full' in m or 'neg' in m:
        return 'full_neg'
    return m.strip()


def _action_from_iou(iou: float):
    if iou is None:
        return None
    if iou == 0:
        return 'reject'
    if iou <= 0.6:
        return 'major revision'
    if iou <= 0.9:
        return 'minor revision'
    return 'accept'


def _parse_prediction(text: str):
    pred_action = None
    pred_mask_type = None
    pred_iou = None
    if text is None:
        return pred_action, pred_mask_type, pred_iou
    action_match = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE | re.DOTALL)
    if action_match:
        pred_action = _normalize_action(action_match.group(1))
    mask_match = re.search(r"<mask_type>(.*?)</mask_type>", text, re.IGNORECASE | re.DOTALL)
    if mask_match:
        pred_mask_type = _normalize_mask_type(mask_match.group(1))
    iou_matches = re.findall(r"<iou>(.*?)</iou>", text, re.IGNORECASE | re.DOTALL)
    parsed_ious = []
    for raw_iou in iou_matches:
        try:
            value = float(raw_iou.strip())
        except Exception:
            continue
        if math.isfinite(value):
            parsed_ious.append(value)
    if parsed_ious:
        first_iou = parsed_ious[0]
        if 0.0 <= first_iou <= 1.0:
            pred_iou = first_iou
        else:
            # Some outputs mention an error rate before the actual IoU.
            valid_ious = [value for value in parsed_ious[1:] if 0.0 <= value <= 1.0]
            if valid_ious:
                pred_iou = valid_ious[-1]
    return pred_action, pred_mask_type, pred_iou




def _calc_prf(counts):
    beta_sq = 4.0  # beta=2
    metrics = {}
    for cls, c in counts.items():
        tp = c['tp']
        fp = c['fp']
        fn = c['fn']
        # 若有样本 (tp+fn>0) 但未命中预测 (tp+fp==0)，precision 视为 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 if (tp + fn) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        # F-beta (beta=2): emphasize recall
        if precision is not None and recall is not None:
            denom = beta_sq * precision + recall
            if denom > 0:
                f_beta = (1 + beta_sq) * precision * recall / denom
            else:
                f_beta = 0.0  # precision=recall=0 的情况
        else:
            f_beta = None
        metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f_beta}
    return metrics


def _calc_macro_prf(counts):
    """宏平均，过滤掉 GT 未出现的类别（tp+fn==0）。用于视频口径。"""
    prf = _calc_prf(counts)
    vals = []
    for cls, c in counts.items():
        # 只统计 GT 中出现过的类别（tp+fn>0），避免无正例的类用 precision=0 拉低均值
        if (c.get('tp', 0) + c.get('fn', 0)) == 0:
            continue
        v = prf.get(cls, {})
        if v.get('precision') is None and v.get('recall') is None and v.get('f1') is None:
            continue
        vals.append(v)
    if len(vals) == 0:
        return {'precision': None, 'recall': None, 'f1': None}
    return {
        'precision': sum(v['precision'] for v in vals if v['precision'] is not None) / len(vals),
        'recall': sum(v['recall'] for v in vals if v['recall'] is not None) / len(vals),
        'f1': sum(v['f1'] for v in vals if v['f1'] is not None) / len(vals),
    }


def _calc_macro_prf_all(prf):
    """宏平均，不额外过滤类别；用于 image 口径（原始行为）。"""
    vals = [v for v in prf.values() if v['precision'] is not None or v['recall'] is not None or v['f1'] is not None]
    if len(vals) == 0:
        return {'precision': None, 'recall': None, 'f1': None}
    return {
        'precision': sum(v['precision'] for v in vals if v['precision'] is not None) / len(vals),
        'recall': sum(v['recall'] for v in vals if v['recall'] is not None) / len(vals),
        'f1': sum(v['f1'] for v in vals if v['f1'] is not None) / len(vals),
    }


def _calc_micro_prf(counts): 
    tp = sum(c['tp'] for c in counts.values())
    fp = sum(c['fp'] for c in counts.values())
    fn = sum(c['fn'] for c in counts.values())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 if (tp + fn) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None:
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    else:
        f1 = None
    return {'precision': precision, 'recall': recall, 'f1': f1}


def _calc_accuracy(counts):
    """Compute accuracy from multi-class TP/FN counts (ignores FP over-counting)."""
    correct = sum(c['tp'] for c in counts.values())
    total = sum(c['tp'] + c['fn'] for c in counts.values())
    return correct / total if total > 0 else None


def _mean_metric_dict(metric_list):
    """对 [{'precision':..,'recall':..,'f1':..}, ...] 按 key 取非空均值。"""
    if len(metric_list) == 0:
        return {'precision': None, 'recall': None, 'f1': None}
    keys = ['precision', 'recall', 'f1']
    out = {}
    for k in keys:
        vals = [m.get(k) for m in metric_list if m.get(k) is not None]
        out[k] = sum(vals) / len(vals) if len(vals) > 0 else None
    return out


def _merge_counts(dst, src):
    """累加两个 count 字典的 tp/fp/fn。"""
    for k, v in src.items():
        if k not in dst:
            dst[k] = {'tp': 0, 'fp': 0, 'fn': 0}
        dst[k]['tp'] += v.get('tp', 0)
        dst[k]['fp'] += v.get('fp', 0)
        dst[k]['fn'] += v.get('fn', 0)


def _filter_counts(counts, allowed_labels):
    """Keep only the counts of allowed labels (e.g., legal actions for a mask type)."""
    return {lbl: counts[lbl] for lbl in allowed_labels if lbl in counts}


def _pred_distribution_for_gt(stat_counts, labels, gt_label, total):
    """
    For a fixed GT group (e.g., all gt=perfect samples), recover how predictions are distributed.
    When gt == pred, the count lives in tp of gt_label; otherwise it appears in fp of the predicted label.
    """
    dist = {}
    for lbl in labels:
        if lbl == gt_label:
            cnt = stat_counts.get(lbl, {}).get('tp', 0)
        else:
            cnt = stat_counts.get(lbl, {}).get('fp', 0)
        dist[lbl] = {
            'count': cnt,
            'ratio': (cnt / total) if total and total > 0 else None,
        }
    return dist


def inference_ref_avs_metrics(
    dataloader,
    ckpt_dir,
    model,
    tokenizer,
    test_name='test_s',
    eval_mode='image',
    compute_dtype=None,
    mask_rank_filter=None,
    mask_type_filter=None,
):
    mask_rank_filter = -1 if mask_rank_filter is None else mask_rank_filter
    mask_type_label = _normalize_mask_type(mask_type_filter) if mask_type_filter is not None else "all"
    if mask_type_label is None or mask_type_label == "all":
        mask_type_label = "all"
    rank_label = "all" if mask_rank_filter == -1 else str(mask_rank_filter)
    save_suffix = f"{eval_mode}_mask-{mask_type_label}_rank-{rank_label}"
    save_dir = join(ckpt_dir, f'{test_name}_{save_suffix}')
    os.makedirs(save_dir, exist_ok=True)
    fp = join(save_dir, f'predictions_{save_suffix}.jsonl')
    fp_video_avg = join(save_dir, f'predictions_video_avg_{save_suffix}.jsonl') if eval_mode == 'video' else None
    if exists(fp):
        os.remove(fp)
    if fp_video_avg is not None and exists(fp_video_avg):
        os.remove(fp_video_avg)
    pbar = tqdm(total=len(dataloader), desc=f'{test_name} {eval_mode} {mask_type_label} {rank_label}')
    
    action_classes = ['accept', 'major revision', 'minor revision', 'reject']
    mask_classes = ['perfect', 'cutout', 'erode', 'dilate', 'merge', 'full_neg', 'null']
    active_mask_classes = mask_classes if mask_type_label == "all" else [mask_type_label]

    legal_actions_map = {
        'perfect': ['accept'],
        'full_neg': ['reject'],
        'null': ['reject'],
        'cutout': ['major revision', 'minor revision'],
        'erode': ['major revision', 'minor revision'],
        'dilate': ['major revision', 'minor revision'],
        'merge': ['major revision', 'minor revision', 'reject'],
    }

    def _legal_actions(mask_type):
        return legal_actions_map.get(mask_type, action_classes)

    def _init_counts(classes):
        return {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in classes}

    def _update_counts(counts, gt_label, pred_label, valid_labels):
        if gt_label in valid_labels:
            if pred_label == gt_label:
                counts[gt_label]['tp'] += 1
            else:
                counts[gt_label]['fn'] += 1
                if pred_label in counts:
                    counts[pred_label]['fp'] += 1
        elif pred_label in valid_labels:
            counts[pred_label]['fp'] += 1

    def _ensure_type_stat(m_type):
        if m_type not in type_stats:
            type_stats[m_type] = {
                'action_counts': _init_counts(action_classes),
                'mask_counts': _init_counts(mask_classes),
                'sq_errors': [],
                'num_samples': 0,
            }
        return type_stats[m_type]

    action_counts = _init_counts(action_classes)
    mask_counts = _init_counts(mask_classes)
    sq_errors = []
    # 按 mask 类型拆分的统计
    type_stats = {}
    for _m in mask_classes:
        _ensure_type_stat(_m)
    all_records = []
    failure_reasons = {
        'action': [],
        'mask_type': [],
        'iou_value': [],
    }

    autocast_dtype = compute_dtype if compute_dtype in (torch.float16, torch.bfloat16) else None

    for _, sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        bs = len(batch_metadata)
        sample = prepare_sample(data=sample, dtype=compute_dtype)
        sample.update(
            {
                'use_cache': True,
                'max_new_tokens': 256,
                'do_sample': False,
                'temperature': 0.0,
            }
        )
        # set_trace()
        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    gen_outputs = model.generate(
                        **sample,
                        return_dict_in_generate=True,
                        output_hidden_states=False,
                    )
            else:
                gen_outputs = model.generate(
                    **sample,
                    return_dict_in_generate=True,
                    output_hidden_states=False,
                )

        sequences = gen_outputs.sequences
        outputs = tokenizer.batch_decode(sequences, skip_special_tokens=False)

        print("--" * 40 )
        print('output:', outputs)

        for i in range(bs):
            # set_trace()

            meta = batch_metadata[i]
            print(meta['mask_path'])
            print("--" * 40 )
            raw_output = outputs[i]
            pred_action, pred_mask_type, pred_iou = _parse_prediction(raw_output)

            gt_iou = meta.get('iou_to_gt')
            gt_mask_type = _normalize_mask_type(meta.get('mask_type'))
            # gt_action = _action_from_iou(gt_iou if gt_iou is not None else None)
            gt_action = meta.get('action')
            allowed_actions = action_classes  # 先全量计数，后续按过滤条件再筛选合法动作

            stat = _ensure_type_stat(gt_mask_type)
            stat['num_samples'] += 1

            if gt_iou is not None:
                effective_pred_iou = pred_iou if pred_iou is not None else 0.0 #! 设置为 0，gt_iou 越大，损失越大
                sq_errors.append((effective_pred_iou - gt_iou) ** 2)
                stat['sq_errors'].append((effective_pred_iou - gt_iou) ** 2)

            _update_counts(action_counts, gt_action, pred_action, allowed_actions)
            _update_counts(mask_counts, gt_mask_type, pred_mask_type, mask_classes) 
            _update_counts(stat['action_counts'], gt_action, pred_action, allowed_actions)
            _update_counts(stat['mask_counts'], gt_mask_type, pred_mask_type, mask_classes)

            # set_trace()
            uid = meta.get('uid')
            if pred_action is None and uid is not None:
                failure_reasons['action'].append(uid)
            if pred_mask_type is None and uid is not None:
                failure_reasons['mask_type'].append(uid)
            if pred_iou is None and uid is not None:
                failure_reasons['iou_value'].append(uid)

            record_to_save = {
                'uid': uid,
                'gt_mask_type': gt_mask_type,
                'pred_mask_type': pred_mask_type,
                'gt_iou': gt_iou,
                'pred_iou': pred_iou,
                'gt_action': gt_action,
                'pred_action': pred_action,
                'raw_output': raw_output,
            }
            write2json(fp=fp, dict_data=record_to_save)
            record_for_video = {
                **record_to_save,
                'vid': meta.get('vid'),
                'frame_idx': meta.get('frame_idx'),
            }
            all_records.append(record_for_video)
        pbar.update(1)
    pbar.close()

    def _calc_rmse_from_errors(errs):
        return math.sqrt(sum(errs) / len(errs)) if len(errs) > 0 else None

    def _calc_acc(hit, tot):
        return hit / tot if tot > 0 else None

    # 仅以当前过滤后的数据出现过的 GT action 作为合法类别
    allowed_actions_global = sorted(
        {
            _normalize_action(rec.get('gt_action'))
            for rec in all_records
            if _normalize_action(rec.get('gt_action')) is not None
        }
    )
    allowed_actions_per_mask = {}
    for rec in all_records:
        ga = _normalize_action(rec.get('gt_action'))
        mt = _normalize_mask_type(rec.get('gt_mask_type'))
        if ga is None or mt is None:
            continue
        allowed_actions_per_mask.setdefault(mt, set()).add(ga)

    rmse = _calc_rmse_from_errors(sq_errors) 
    action_labels_for_metrics = allowed_actions_global if mask_type_label == "all" else sorted(allowed_actions_per_mask.get(mask_type_label, set()))
    filtered_action_counts = _filter_counts(action_counts, action_labels_for_metrics if len(action_labels_for_metrics) > 0 else action_classes)
    action_metrics = _calc_prf(filtered_action_counts) 
    filtered_mask_counts = _filter_counts(mask_counts, active_mask_classes)
    mask_metrics = _calc_prf(filtered_mask_counts)
    action_micro = _calc_micro_prf(filtered_action_counts)
    mask_micro = _calc_micro_prf(filtered_mask_counts)
    action_macro = _calc_macro_prf(filtered_action_counts) if eval_mode == 'video' else _calc_macro_prf_all(_calc_prf(filtered_action_counts))
    mask_macro = _calc_macro_prf(filtered_mask_counts) if eval_mode == 'video' else _calc_macro_prf_all(_calc_prf(filtered_mask_counts))

    rank_msg = "all ranks (average across mask candidates)" if mask_rank_filter == -1 else f"rank={mask_rank_filter}"
    print(f"[ref-avs] mask_rank_filter: {rank_msg}")
    print(f"[ref-avs] samples: {len(dataloader.dataset)}")
    print(f"[ref-avs] RMSE (IoU): {rmse}") 
    print(f"[ref-avs] Action metrics: {action_metrics}") # 
    print(f"[ref-avs] Action micro: {action_micro}") 
    print(f"[ref-avs] Action macro: {action_macro}")
    print(f"[ref-avs] Mask metrics: {mask_metrics}") # 
    print(f"[ref-avs] Mask micro: {mask_micro}") 
    print(f"[ref-avs] Mask macro: {mask_macro}")
    print(f"[ref-avs] prediction failures: "
          f"action={len(failure_reasons['action'])}, "
          f"mask_type={len(failure_reasons['mask_type'])}, "
          f"iou_value={len(failure_reasons['iou_value'])}")
    metrics_summary = {
        'mask_rank_filter': mask_rank_filter,
        'mask_rank_desc': rank_msg,
        'mask_type_filter': mask_type_label,
        'eval_mode': eval_mode,
        'samples': len(dataloader.dataset),
        'rmse': rmse,
        'action_metrics': action_metrics,
        'action_micro': action_micro,
        'action_macro': action_macro,
        'mask_metrics': mask_metrics,
        'mask_micro': mask_micro,
        'mask_macro': mask_macro,
        'prediction_failures': {
            'action': {
                'count': len(failure_reasons['action']),
                'uids': failure_reasons['action'],
            },
            'mask_type': {
                'count': len(failure_reasons['mask_type']),
                'uids': failure_reasons['mask_type'],
            },
            'iou_value': {
                'count': len(failure_reasons['iou_value']),
                'uids': failure_reasons['iou_value'],
            },
        },
        'per_mask': {},
    }
    for m_type in active_mask_classes:
        stat = _ensure_type_stat(m_type)
        rmse_t = _calc_rmse_from_errors(stat['sq_errors'])
        mask_prf_map = _calc_prf(stat['mask_counts'])
        mask_metrics_t = mask_prf_map.get(m_type, {'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
        allowed_actions_t = sorted(allowed_actions_per_mask.get(m_type, allowed_actions_global))
        filtered_action_counts_t = _filter_counts(stat['action_counts'], allowed_actions_t if len(allowed_actions_t) > 0 else action_classes)
        action_metrics_t = _calc_macro_prf(filtered_action_counts_t) if eval_mode == 'video' else _calc_macro_prf_all(_calc_prf(filtered_action_counts_t))
        print(f"[ref-avs][{m_type}] nums: {stat['num_samples']} RMSE: {rmse_t} mask: {mask_metrics_t} action: {action_metrics_t}")
        metrics_summary['per_mask'][m_type] = {
            'num_samples': stat['num_samples'],
            'rmse': rmse_t,
            'mask_metrics': mask_metrics_t,
            'action_metrics': action_metrics_t,
        }
    # Confusion-like breakdown when evaluating a single mask type
    if mask_type_label != "all":
        sub_records = [r for r in all_records if _normalize_mask_type(r.get('gt_mask_type')) == mask_type_label]
        total_single = len(sub_records)

        def _build_distribution(records, key, norm_fn=None):
            counts = {}
            for rec in records:
                val = rec.get(key)
                val = norm_fn(val) if norm_fn is not None else val
                counts[val] = counts.get(val, 0) + 1
            return {
                lbl: {
                    'count': cnt,
                    'ratio': (cnt / total_single) if total_single > 0 else None,
                }
                for lbl, cnt in counts.items()
            }

        mask_pred_dist = _build_distribution(sub_records, 'pred_mask_type', _normalize_mask_type)
        action_pred_dist = _build_distribution(sub_records, 'pred_action', _normalize_action)
        print(f"[ref-avs][{mask_type_label}] pred mask distribution: {mask_pred_dist}")
        print(f"[ref-avs][{mask_type_label}] pred action distribution: {action_pred_dist}")
        metrics_summary['gt_mask_type_breakdown'] = {
            'gt_mask_type': mask_type_label,
            'total_samples': total_single,
            'pred_mask_distribution': mask_pred_dist,
            'pred_action_distribution': action_pred_dist,
        }

    # set_trace()

    # video 模式下，再汇总每个视频的 10 帧平均结果
    if eval_mode == 'video' and len(all_records) > 0:
        vid_groups = {}
        for rec in all_records:
            # vid = rec.get('vid')
            vid = rec.get('uid') 
            if vid not in vid_groups:
                vid_groups[vid] = []
            vid_groups[vid].append(rec)
        # set_trace()
        avg_records = []
        # 按视频累计（逐帧统计，不再投票）
        action_acc_list, mask_acc_list = [], []
        action_macro_list, mask_macro_list = [], []
        type_stats_vid = {
            m: {
                'action_counts': {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in action_classes},
                'mask_counts': {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in mask_classes},
                'sq_errors': [],
                'num_samples': 0,
            }
            for m in mask_classes
        }
        sq_errors_vid = []
        for vid, recs in vid_groups.items():
            print(f"==> lenth of uid {vid} is {len(recs)}")
            # assert len(recs) == 10
            # 平均 IoU
            pred_ious = [r['pred_iou'] for r in recs if r.get('pred_iou') is not None]
            gt_ious = [r['gt_iou'] for r in recs if r.get('gt_iou') is not None]
            avg_pred_iou = sum(pred_ious) / len(pred_ious) if len(pred_ious) > 0 else None
            avg_gt_iou = sum(gt_ious) / len(gt_ious) if len(gt_ious) > 0 else None

            counts_action_single = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in action_classes}
            counts_mask_single = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in mask_classes}
            for r in recs:
                ga = _normalize_action(r.get('gt_action'))
                pa = _normalize_action(r.get('pred_action'))
                gm = _normalize_mask_type(r.get('gt_mask_type'))
                pm = _normalize_mask_type(r.get('pred_mask_type'))
                allowed_actions_vid = action_classes
                _update_counts(counts_action_single, ga, pa, allowed_actions_vid)
                _update_counts(counts_mask_single, gm, pm, mask_classes)
                stat_vid = type_stats_vid.get(gm)
                if stat_vid is None:
                    stat_vid = type_stats_vid.setdefault(
                        gm,
                        {
                            'action_counts': {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in action_classes},
                            'mask_counts': {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in mask_classes},
                            'sq_errors': [],
                            'num_samples': 0,
                        },
                    )
                stat_vid['num_samples'] += 1
                _update_counts(stat_vid['action_counts'], ga, pa, allowed_actions_vid)
                _update_counts(stat_vid['mask_counts'], gm, pm, mask_classes)

            if avg_pred_iou is not None and avg_gt_iou is not None:
                sq_errors_vid.append((avg_pred_iou - avg_gt_iou) ** 2)

            action_acc_single = _calc_accuracy(counts_action_single)
            mask_acc_single = _calc_accuracy(counts_mask_single)
            if action_acc_single is not None:
                action_acc_list.append(action_acc_single)
            if mask_acc_single is not None:
                mask_acc_list.append(mask_acc_single)
            action_macro_single = _calc_macro_prf(counts_action_single)
            mask_macro_single = _calc_macro_prf(counts_mask_single)
            action_macro_list.append(action_macro_single)
            mask_macro_list.append(mask_macro_single)

            avg_record = {
                'vid': vid,
                'avg_pred_iou': avg_pred_iou,
                'avg_gt_iou': avg_gt_iou,
                'action_counts': counts_action_single,
                'mask_counts': counts_mask_single,
                'action_acc': action_acc_single,
                'mask_acc': mask_acc_single,
                'action_macro': action_macro_single,
                'mask_macro': mask_macro_single,
                'gt_actions': [ _normalize_action(r.get('gt_action')) for r in recs ],
                'pred_actions': [ _normalize_action(r.get('pred_action')) for r in recs ],
                'gt_mask_types': [ _normalize_mask_type(r.get('gt_mask_type')) for r in recs ],
                'pred_mask_types': [ _normalize_mask_type(r.get('pred_mask_type')) for r in recs ],
                'frame_num': len(recs),
            }
            if fp_video_avg is not None:
                write2json(fp=fp_video_avg, dict_data=avg_record)
            avg_records.append(avg_record)

        rmse_vid = math.sqrt(sum(sq_errors_vid) / len(sq_errors_vid)) if len(sq_errors_vid) > 0 else None
        action_acc_vid_avg = sum(action_acc_list) / len(action_acc_list) if len(action_acc_list) > 0 else None
        mask_acc_vid_avg = sum(mask_acc_list) / len(mask_acc_list) if len(mask_acc_list) > 0 else None
        action_macro_vid_avg = _mean_metric_dict(action_macro_list)
        mask_macro_vid_avg = _mean_metric_dict(mask_macro_list)
        print(f"[ref-avs][video-avg] videos: {len(avg_records)}")
        print(f"[ref-avs][video-avg] RMSE (IoU): {rmse_vid}") # total #! 汇报这个
        print(f"[ref-avs][video-avg] Action acc (avg per video): {action_acc_vid_avg}")
        print(f"[ref-avs][video-avg] Action macro (avg per video): {action_macro_vid_avg}")
        print(f"[ref-avs][video-avg] Mask acc (avg per video): {mask_acc_vid_avg}")
        print(f"[ref-avs][video-avg] Mask macro (avg per video): {mask_macro_vid_avg}")
        metrics_summary['video_avg'] = {
            'videos': len(avg_records),
            'rmse': rmse_vid,
            'action_acc_avg_per_video': action_acc_vid_avg,
            'mask_acc_avg_per_video': mask_acc_vid_avg,
            'action_macro_avg_per_video': action_macro_vid_avg,
            'mask_macro_avg_per_video': mask_macro_vid_avg,
        }
    # set_trace()
    metrics_fp = join(save_dir, f'metrics_summary_{save_suffix}.json')
    with open(metrics_fp, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[ref-avs] metrics saved to {metrics_fp}")


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InferenceArguments))
    model_args, data_args, training_args, infer_args = parser.parse_args_into_dataclasses()

    if model_args.llm_name == 'llama':
        d_model = 4096
    elif model_args.llm_name == 'qwen':
        d_model = 3584

    local_rank = training_args.local_rank
    compute_dtype = torch.float32
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16
    
    pretrain_model_name_or_path = model_args.model_name_or_path
    if model_args.llm_name == 'llama':
        from models.unified_llama import UnifiedForCausalLM
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config=config,
            torch_dtype=compute_dtype
        )
    elif model_args.llm_name == 'qwen':
        from models.unified_qwen import UnifiedForCausalLM
        from transformers import Qwen2Config
        config = Qwen2Config.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config = config,
            torch_dtype = compute_dtype
        )

    model.config.use_cache = True # 推理时开启

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        if training_args.use_hyper_lora: # False
            from peft_hyper import LoraConfig,get_peft_model
            lora_trainable="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
            target_modules = lora_trainable.split(',')
            lora_rank = 8
            lora_alpha = 16
            lora_dropout = 0.05
            lora_nums = 3
            modules_to_save = None
            peft_config = LoraConfig(
                task_type = "CAUSAL_LM",
                target_modules = target_modules,
                inference_mode = False,
                r = lora_rank, 
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                lora_nums = lora_nums,
                # modules_to_save=modules_to_save
            )
            model = get_peft_model(model, peft_config)
        else:
            from peft import LoraConfig, get_peft_model
            #! add two lines
            lora_trainable = "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
            target_modules = lora_trainable.split(',')

            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                # target_modules=find_all_linear_names(model),
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)


    if model_args.llm_name == 'qwen':
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
    
    elif model_args.llm_name == 'llama':
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ori_tokenizer_vocab_nums = len(tokenizer)
    model.get_model().pad_token_id = tokenizer.pad_token_id
    model.get_model().init_multimodal_modules(visual_branch=training_args.visual_branch,
                                              audio_branch=training_args.audio_branch,
                                              segment_branch=training_args.seg_branch,
                                              d_model=d_model,vit_ckpt_path=model_args.vit_ckpt_path,
                                              select_layer_list=model_args.select_layer_list,
                                              select_feature=model_args.select_feature,
                                              image_size=model_args.image_size,
                                              patch_size=model_args.patch_size,
                                              visual_query_token_nums=model_args.visual_query_token_nums,
                                              audio_query_token_nums=model_args.audio_query_token_nums,
                                              BEATs_ckpt_path=model_args.BEATs_ckpt_path,
                                              prompt_embed_dim=model_args.prompt_embed_dim,
                                              mask_decoder_transformer_depth=model_args.mask_decoder_transformer_depth,
                                              low_res_mask_size=model_args.low_res_mask_size,
                                              avs_query_num=model_args.avs_query_num,
                                              num_classes=model_args.num_classes,
                                              query_generator_num_layers=model_args.query_generator_num_layers,
                                              dice_loss_weight=training_args.dice_loss_weight,
                                              bce_loss_weight=training_args.bce_loss_weight,
                                              use_vqgan=False)

    model.initialize_MM_tokenizer(tokenizer, use_vqgan=False)
    MM_tokenizer_vocab_nums = len(tokenizer)
    print('ori_tokenizer_vocab_nums: ',ori_tokenizer_vocab_nums, ' MM_tokenizer_vocab_nums: ',MM_tokenizer_vocab_nums)

    # set_trace()

    infer_avs = False
    ckpt_dir = infer_args.ckpt_dir
    nonlora_ckpt_dir = infer_args.nonlora_ckpt_dir
    if not infer_avs:
        ckpt_path = join(ckpt_dir,'finetune_weights.bin')
        # ckpt = torch.load(ckpt_path,map_location='cpu')
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt,strict=False)
        print(f'load ckpt from {ckpt_path} finished...')

        nolora_ckpt_path = join(nonlora_ckpt_dir,'non_lora_trainables.bin')
        nolora_ckpt = torch.load(nolora_ckpt_path,map_location='cpu')
        # set_trace()
        model.load_state_dict(nolora_ckpt,strict=False)
        print(f'load ckpt from {nolora_ckpt_path} finished...')
   

    device = infer_args.device
    torch.cuda.set_device(device)
    # model.npu()
    model.to(device)
    model.eval()
    if training_args.bf16:
        model.to(torch.bfloat16)
    
    image_processor = model.get_model().visual_encoder.image_processor if training_args.visual_branch else None
    dataset, collator = get_dataset_collator(data_args=data_args, tokenizer=tokenizer, 
                                             image_processor=image_processor,mode='test',
                                             test_name=infer_args.test_name)
    
    batch_size = 1 if (infer_avs or data_args.ref_avs_task) else 8 
    # batch_size = 1
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False,collate_fn=collator,drop_last=False)
    
    print("--" * 40 )
    print(infer_args.ckpt_dir)
    print("--" * 40 )

    if data_args.ref_avs_task:
        test_name = infer_args.test_name
        eval_mode = getattr(data_args, "refavs_eval_mode", "image")
        inference_ref_avs_metrics(
            dataloader=dataloader,
            ckpt_dir=ckpt_dir,
            model=model,
            tokenizer=tokenizer,
            test_name=test_name,
            eval_mode=eval_mode,
            compute_dtype=compute_dtype,
            mask_rank_filter=getattr(data_args, "refavs_mask_rank_filter", -1),
            mask_type_filter=getattr(data_args, "refavs_mask_type_filter", "all"),
        )

    print('inference finished...')
    print("--" * 40 )
    print(infer_args.ckpt_dir)
    print("--" * 40 )
    
if __name__ == "__main__":
    train()
