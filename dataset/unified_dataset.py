import json
import ast
import os
from os.path import join,exists
import numpy as np
import pandas as pd
import cv2,csv
from typing import Sequence,Dict
from dataclasses import dataclass
import librosa
from PIL import Image
import torch
import random
import transformers
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from decord import VideoReader
from transformers import CLIPImageProcessor

from dataset.audio_processor import preprocess
from ipdb import set_trace


class UnifiedDataset(Dataset):
    def __init__(
        self,
        mode='train', # train,val,test
        video_processor: CLIPImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        data_args=None,
        image_size = 224,
        video_frame_nums = 10,
        ref_avs_task = False,
        multi_frames = False,
    ) -> None:
        super().__init__()

        self.mode=mode
        self.video_processor = video_processor
        self.multi_frames = multi_frames
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums
        self.data_args = data_args

        self.samples = []
        self.tot = 0

        if ref_avs_task:
            self.add_ref_avs_samples()

        
        print(f'tot training sample nums: {self.tot}')


    def add_ref_avs_samples(self):
        """
        读取 ref-avs 训练元数据：
        1）从 meta csv 里获取 uid/kfid/表达式等基础信息；
        2）再从 cot json 里拿到 gt、part neg、full neg 等不同类型的 mask 以及对应 response/iou；
        3）将关键帧路径、mask 候选列表、音频与视频帧路径打包到样本，后续在 collate 阶段按需平衡正负。
        """
        data_root = getattr(self.data_args, "refavs_data_root", "../MQ-RAVSBench")
        cot_anno_root = getattr(
            self.data_args,
            "refavs_cot_json_path",
            "../MQ-RAVSBench/train_test_meta_files/train_audit_only_filtered.json",
        )
        meta_csv_path = getattr(
            self.data_args,
            "refavs_meta_csv_path",
            "../MQ-RAVSBench/train_test_meta_files/metadata.csv",
        )

        with open(cot_anno_root, 'r') as f:
            cot_anno_lines = json.load(f)

        def _inject_iou_value(response: str, iou_val):
            """将占位符 <iou_token> 替换为真实数值包装的 <iou>...</iou>。"""
            if response is None:
                return ''
            try:
                iou_str = f"{float(iou_val):.4f}"
            except Exception:
                iou_str = ""
            replacement = f"<iou> {iou_str} </iou>"
            if "<iou_token>" in response:
                return response.replace("<iou_token>", replacement)
            return response

        def _resolve_path(path_str: str):
            """将 json 中相对 MQ-RAVSBench 的路径转成可读取路径."""
            if path_str is None:
                return None
            if os.path.isabs(path_str):
                return path_str
            return join(data_root, path_str)

        tot = 0
        with open(meta_csv_path, 'r') as f:
            rows = csv.reader(f)
            for row in rows:
                vid, uid, split, fid, exp, kfid = row
                if split != 'train':
                    continue
                vid = uid.rsplit('_', 2)[0]  # TODO: use encoded id.
                kfid = int(kfid)

                # 处理音频
                if uid.startswith('null_'):  # 判断是否为静音音频
                    audio_path = join(data_root, 'media_cross', vid, 'audio.wav')
                else:
                    audio_path = join(data_root, 'media', vid, 'audio.wav')

                # 视频全帧 & 关键帧路径
                image_path_list = [join(data_root, 'media', vid, 'frames', f"{i}.jpg") for i in range(10)]
                key_frame_path = join(data_root, 'media', vid, 'frames', f"{kfid}.jpg")

                # json 中的多种 mask 候选及响应
                cot_entry = cot_anno_lines.get(uid, {})
                mask_candidates = []
                gt_entry = cot_entry.get('gt_mask_path', {})
                if gt_entry:
                    mask_candidates.append(
                        {
                            'mask_type': 'perfect',
                            'mask_path': _resolve_path(gt_entry.get('mask_path')),
                            'response': _inject_iou_value(gt_entry.get('response', ''), gt_entry.get('iou_to_gt', 1.0)),
                            'iou_to_gt': gt_entry.get('iou_to_gt', 1.0),
                        }
                    )

                part_neg_masks = cot_entry.get('part_neg_masks', {})
                for m_type, items in part_neg_masks.items():
                    for item in items:
                        mask_candidates.append(
                            {
                                'mask_type': m_type,
                                'mask_path': _resolve_path(item.get('mask_path')),
                                'response': _inject_iou_value(item.get('response', ''), item.get('iou_to_gt', 0.0)),
                                'iou_to_gt': item.get('iou_to_gt', 0.0),
                            }
                        )

                full_neg_masks = cot_entry.get('full_neg_masks', [])
                for item in full_neg_masks:
                    mask_candidates.append(
                        {
                            'mask_type': 'full_neg',
                            'mask_path': _resolve_path(item.get('mask_path')),
                            'response': _inject_iou_value(item.get('response', ''), item.get('iou_to_gt', 0.0)),
                            'iou_to_gt': item.get('iou_to_gt', 0.0),
                        }
                    )

                null_masks = cot_entry.get('null_masks', [])
                for item in null_masks:
                    mask_candidates.append(
                        {
                            'mask_type': 'null',
                            'mask_path': _resolve_path(item.get('mask_path')),
                            'response': _inject_iou_value(item.get('response', ''), item.get('iou_to_gt', 0.0)),
                            'iou_to_gt': item.get('iou_to_gt', 0.0),
                        }
                    )

                instruction = (
                    "video:\n<video_start><video><video_end>\n"
                    "audio:\n<audio_start><audio><audio_end>\n"
                    f"Given the referential expression: '{exp.lower()}', "
                    "the key frame image <image_start><image><image_end>, and its corresponding segmentation mask <mask_start><mask><mask_end>, "
                    "analyze all cues to audit the mask's quality.\n"
                )

                self.samples.append(
                    {
                        'instruction': instruction,
                        # output 在 collate 阶段依据选中的 mask 响应再写入
                        'output': '',
                        'audio_path': audio_path,
                        'image_path_list': image_path_list,
                        'key_frame_path': key_frame_path,
                        'mask_candidates': mask_candidates,
                        'vid': vid,
                        'uid': uid,
                        'fid': fid,
                        'kfid': kfid,
                        'exp': exp,
                        'task_name': 'ref-avs',
                    }
                )
                tot += 1

        self.tot += tot
        print(f'ref-avs sample nums: {tot}')


    def __len__(self):
        return len(self.samples) 


    def __getitem__(self,idx):
        sample = self.samples[idx]
        task_name = sample['task_name']
        instruction = sample['instruction']
        output = sample.get('output')


        data = {
            'instruction': "<s>" +instruction,
            'output':output + "</s>",
            'task_name':task_name,
        }
   
        if task_name == 'ref-avs':
            ## video
            image_path_list = sample['image_path_list']
            video = []
            for path in image_path_list:
                image = Image.open(path).convert('RGB')
                image = image.resize((224,224))
                image = self.video_processor.preprocess([image],return_tensors='pt')
                image = image['pixel_values']   # [1, c, h, w]
                video.append(image)
            video = torch.cat(video,dim=0) # t,c,h,w
            data['video'] = video

            ## audio
            audio_path = sample['audio_path']
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 10
            nums_per_second = int(length / tot)
            indices = [i for i in range(tot)]
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if len(audio_seg) < 1 * nums_per_second:
                    sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['audio'] = audio_feature

            ## 关键帧图像，用于替换指令中的 <image> 占位符
            key_frame_path = sample.get('key_frame_path')
            if key_frame_path is not None and exists(key_frame_path):
                key_frame = Image.open(key_frame_path).convert('RGB')
                key_frame = key_frame.resize((224,224))
                key_frame = self.video_processor.preprocess([key_frame],return_tensors='pt')
                key_frame = key_frame['pixel_values']  # t,c,h,w
                data['image'] = key_frame
                data['key_frame_path'] = key_frame_path

            ## mask 候选信息保留到样本里，后续 collate 时再按类型选择、平衡正负
            data['mask_candidates'] = sample.get('mask_candidates', [])
            data['uid'] = sample.get('uid')
            data['vid'] = sample.get('vid')
            data['exp'] = sample.get('exp')
            data['kfid'] = sample.get('kfid')

            ## image
            # image_path = sample['image_path']
            # image = Image.open(image_path).convert('RGB')
            # image = image.resize((224,224))
            # image = self.video_processor.preprocess([image],return_tensors='pt')
            # image = image['pixel_values']  # t,c,h,w
            # data['image'] = image
            
            ## mask
            # mask_path = sample['mask_path']
            # mask = cv2.imread(mask_path)
            # gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            # gt_mask = gray_mask > 0
            # gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
            # gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
            # data['mask'] = gt_mask

        return data


class UnifiedTestDataset(Dataset):
    def __init__(
        self,
        mode='test', # train,val,test
        video_processor: CLIPImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        data_args=None,
        image_size = 224,
        video_frame_nums = 10,
        multi_frames = False,
        ref_avs_task = True,
        test_name = 'test_s',  # for ref-avs: test_s, test_u
    ) -> None:
        super().__init__()

        self.mode=mode
        self.video_processor = video_processor
        self.multi_frames = multi_frames
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums
        self.test_name = test_name
        self.data_args = data_args
        self.eval_mode = getattr(self.data_args, "refavs_eval_mode", "image") if data_args is not None else "image"
        self.mask_type_filter = getattr(self.data_args, "refavs_mask_type_filter", "all") if data_args is not None else "all"
        self.mask_rank_filter = getattr(self.data_args, "refavs_mask_rank_filter", -1) if data_args is not None else -1

        self.samples = []
        self.tot = 0
        
        if ref_avs_task:
            self.add_ref_avs_samples()
        print(f'tot test sample nums: {self.tot}')


    def add_ref_avs_samples(self):
        eval_mode = getattr(self.data_args, "refavs_eval_mode", "image") if self.data_args is not None else "image"
        self.eval_mode = eval_mode
        data_root = getattr(self.data_args, "refavs_data_root", "../MQ-RAVSBench")
        meta_csv_path = getattr(
            self.data_args,
            "refavs_meta_csv_path",
            "../MQ-RAVSBench/train_test_meta_files/metadata.csv",
        )
        # 按 test_name 选择 test_s/test_u 的 JSON
        if eval_mode == 'image':
            if self.test_name.startswith('test_u'):
                test_image_json = getattr(
                    self.data_args,
                    "refavs_test_u_image_json_path",
                    "../MQ-RAVSBench/train_test_meta_files/test_u_image_filtered.json",
                )
            else:
                test_image_json = getattr(
                    self.data_args,
                    "refavs_test_image_json_path",
                    "../MQ-RAVSBench/train_test_meta_files/test_s_image_filtered.json",
                )
            test_video_json = None
        else:
            if self.test_name.startswith('test_u'):
                test_video_json = getattr(
                    self.data_args,
                    "refavs_test_u_video_json_path",
                    "../MQ-RAVSBench/train_test_meta_files/test_u_video_filtered.json",
                )
            else:
                test_video_json = getattr(
                    self.data_args,
                    "refavs_test_video_json_path",
                    "../MQ-RAVSBench/train_test_meta_files/test_s_video_filtered.json",
                )
            test_image_json = None

        def _resolve_path(path_str: str):
            """统一转换 JSON 中相对 MQ-RAVSBench 的路径。"""
            if path_str is None:
                return None
            if os.path.isabs(path_str):
                return path_str
            return join(data_root, path_str)

        def _normalize_mask_type_name(mask_type: str):
            if mask_type is None:
                return None
            m = mask_type.lower()
            if m in ['gt', 'gt_mask', 'perfect']:
                return 'perfect'
            if 'cut' in m:
                return 'cutout'
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

        mask_type_filter = _normalize_mask_type_name(self.mask_type_filter)
        if mask_type_filter == 'all':
            mask_type_filter = None

        def _load_meta_for_split(target_split: str):
            """从 meta CSV 读取指定 split（test_s/test_u）的 uid 列表及附加信息。"""
            uid2meta = {}
            if meta_csv_path is None or not exists(meta_csv_path):
                return uid2meta
            with open(meta_csv_path, 'r') as f:
                rows = csv.reader(f)
                for row in rows:
                    if len(row) < 6:
                        continue
                    vid, uid, split, fid, exp, kfid = row[:6]
                    if split != target_split:
                        continue
                    try:
                        kfid_val = int(kfid)
                    except Exception:
                        kfid_val = None
                    try:
                        fid_val = int(fid)
                    except Exception:
                        fid_val = None
                    vid_val = vid if vid not in [None, ''] else uid.rsplit('_', 2)[0]
                    uid2meta[uid] = {
                        'vid': vid_val,
                        'exp': exp,
                        'kfid': kfid_val,
                        'fid': fid_val,
                    }
            return uid2meta

        uid2meta = _load_meta_for_split(self.test_name)

        def _iter_json_items(obj):
            """兼容 JSON 顶层为 list 或 {uid: sample} 两种结构。"""
            if isinstance(obj, dict):
                return obj.values()
            if isinstance(obj, list):
                return obj
            return []

        def _get_audio_path(uid: str, vid: str):
            """根据 uid 选择静音/正常音频路径。"""
            prefix = 'media_cross' if uid.startswith('null_') else 'media'
            return join(data_root, prefix, vid, 'audio.wav')

        def _gather_candidates(gt_entry, part_neg_masks, full_neg_masks, null_masks=None):
            candidates = []
            if gt_entry:
                gt_path = gt_entry.get('mask_path') if isinstance(gt_entry, dict) else gt_entry
                candidates.append(
                    {
                        'mask_type': 'perfect',
                        'mask_path': _resolve_path(gt_path),
                        'iou_to_gt': 1.0,
                        'action': 'accept'
                    }
                )
            for m_type, items in part_neg_masks.items():
                for item in items:
                    candidates.append(
                        {
                            'mask_type': m_type,
                            'mask_path': _resolve_path(item.get('mask_path')),
                            'iou_to_gt': item.get('iou_to_gt', 0.0),
                            'rank': item.get('rank'),
                            'action': item.get('action')
                        }
                    )
            for item in full_neg_masks:
                candidates.append(
                    {
                        'mask_type': 'full_neg',
                        'mask_path': _resolve_path(item.get('mask_path')),
                        'iou_to_gt': item.get('iou_to_gt', 0.0),
                        'rank': item.get('rank'),
                        'action': 'reject'
                    }
                )
            for item in null_masks or []:
                candidates.append(
                    {
                        'mask_type': 'null',
                        'mask_path': _resolve_path(item.get('mask_path')),
                        'iou_to_gt': item.get('iou_to_gt', 0.0),
                        'rank': item.get('rank'),
                        'action': item.get('action', 'reject')
                    }
                )
            # rank 过滤仅对 cutout/erode/dilate 有效
            if self.mask_rank_filter is not None and self.mask_rank_filter > 0:
                filtered = []
                for c in candidates:
                    m_type = _normalize_mask_type_name(c.get('mask_type'))
                    rank_val = c.get('rank')
                    try:
                        rank_val = int(rank_val)
                    except Exception:
                        rank_val = None
                    # if m_type in ['cutout', 'erode', 'dilate']:
                    if m_type in ['cutout', 'erode', 'dilate', 'merge', 'full_neg', 'null']: #! video based test时，都可以选
                        if rank_val == self.mask_rank_filter:
                            filtered.append(c)
                    else:
                        filtered.append(c)
                candidates = filtered
            return candidates

        tot = 0
        if eval_mode == 'image':
            with open(test_image_json, 'r') as f:
                samples_raw = json.load(f)
                # print("lenth samples_raw", len(samples_raw))
            for item in _iter_json_items(samples_raw):
                uid = item.get('uid')
                # print('uid', uid)
                # print('uid', uid not in uid2meta)
                if uid not in uid2meta: # uid2meta 是指定 split（test_s/test_u）的 uid 列表及附加信息
                    continue
                meta = uid2meta[uid]
                vid = meta.get('vid', item.get('vid'))
                exp = meta.get('exp')
                if exp in [None, '']:
                    exp = item.get('exp', '')
                kfid = meta.get('kfid')
                if kfid is None:
                    kfid = int(item.get('kfid', item.get('frame', 0)))
                image_path = _resolve_path(item.get('image_path'))
                # image 模式也要提供完整 10 帧的视频上下文，用于 <video> token
                image_path_list = [
                    join(data_root, 'media', vid, 'frames', f"{i}.jpg") for i in range(self.video_frame_nums)
                ]
                audio_path = _get_audio_path(uid, vid)
                gt_entry = item.get('gt_mask_path', {})
                part_neg_masks = item.get('part_neg_masks', {})
                full_neg_masks = item.get('full_neg_masks', [])
                null_masks = item.get('null_masks', [])
                candidates = _gather_candidates(gt_entry, part_neg_masks, full_neg_masks, null_masks) # 如果有rank—filter，已经通过rank过滤了，这里只要过滤mask type就行
                # print('==> candidates:\n', candidates)
                if mask_type_filter is not None:
                    candidates = [c for c in candidates if _normalize_mask_type_name(c['mask_type']) == mask_type_filter]
                    # print('==> candidates after mask_type_filter:\n', candidates)
                    # print('==> lenth candidates after mask_type_filter:\n', len(candidates))
                if len(candidates) == 0:
                    continue
                instruction = (
                    "video:\n<video_start><video><video_end>\n"
                    "audio:\n<audio_start><audio><audio_end>\n"
                    f"Given the referential expression: '{exp.lower()}', the key frame <image_start><image><image_end> "
                    "and its segmentation mask <mask_start><mask><mask_end>, please audit the mask quality.\n"
                )
                for cand in candidates:
                    self.samples.append(
                        {
                            'instruction': instruction,
                            'output': '',
                            'image_path': image_path,
                            'image_path_list': image_path_list,
                            'audio_path': audio_path,
                            'mask_path': cand['mask_path'],
                            'mask_type': cand['mask_type'],
                            'iou_to_gt': cand['iou_to_gt'],
                            'action': cand['action'],
                            'task_name': 'ref-avs',
                            'uid': uid,
                            'vid': vid,
                            'kfid': kfid,
                            'exp': exp,
                            'ref': exp,
                            'frame_idx': meta.get('fid'),
                        }
                    )
                    tot += 1
        else: # eval_mode ='video'
            with open(test_video_json, 'r') as f:
                videos_raw = json.load(f)
            for video_item in _iter_json_items(videos_raw):
                uid = video_item.get('uid')
                if uid not in uid2meta:
                    continue
                meta = uid2meta[uid]
                vid = meta.get('vid', video_item.get('vid'))
                exp = meta.get('exp')
                if exp in [None, '']:
                    exp = video_item.get('exp', '')
                kfid = meta.get('kfid')
                if kfid is None:
                    kfid = int(video_item.get('kfid', 0))
                frames = video_item.get('frames', [])
                image_path_list = [_resolve_path(f['image_path']) for f in frames]
                audio_path = _get_audio_path(uid, vid)
                # 获取本视频可用的 mask 类型集合
                available_types = set()
                frame_candidates_by_type = []
                for frame in frames:
                    gt_entry = frame.get('gt_mask_path', {})
                    part_neg_masks = frame.get('part_neg_masks', {})
                    full_neg_masks = frame.get('full_neg_masks', [])
                    null_masks = frame.get('null_masks', [])
                    cands = _gather_candidates(gt_entry, part_neg_masks, full_neg_masks, null_masks)
                    norm_cands = [] # 当前 frame 的所有 mask 类型
                    for c in cands:
                        c_norm = c.copy()
                        c_norm['mask_type'] = _normalize_mask_type_name(c.get('mask_type'))
                        available_types.add(c_norm['mask_type'])
                        norm_cands.append(c_norm)
                    frame_candidates_by_type.append(norm_cands) # 所有 frame 的信息，lenth=10

                # print('==> frame_candidates_by_type:\n', frame_candidates_by_type)
                # print('==> lenth frame_candidates_by_type:\n', len(frame_candidates_by_type))

                if mask_type_filter is not None:
                    mask_types_for_eval = [mask_type_filter]
                else:
                    mask_types_for_eval = sorted([t for t in available_types if t is not None])

                for m_type in mask_types_for_eval:
                    for frame, cand_list in zip(frames, frame_candidates_by_type): # 每一帧，及对应的所有类型mask类型信息
                        for cand in cand_list:
                            if cand.get('mask_type') != m_type:
                                continue
                            image_path = _resolve_path(frame['image_path'])
                            instruction = (
                                "video:\n<video_start><video><video_end>\n"
                                "audio:\n<audio_start><audio><audio_end>\n"
                                f"Given the referential expression: '{exp.lower()}', the frame <image_start><image><image_end> "
                                "and its segmentation mask <mask_start><mask><mask_end>, please audit the mask quality.\n"
                            )
                            self.samples.append(
                                {
                                    'instruction': instruction,
                                    'output': '',
                                    'image_path': image_path,
                                    'image_path_list': image_path_list,
                                    'audio_path': audio_path,
                                    'mask_path': cand['mask_path'],
                                    'mask_type': cand['mask_type'],
                                    'iou_to_gt': cand['iou_to_gt'],
                                    'action': cand['action'],
                                    'task_name': 'ref-avs',
                                    'uid': uid,
                                    'vid': vid,
                                    'kfid': kfid,
                                    'exp': exp,
                                    'ref': exp,
                                    'frame_idx': frame.get('frame'),
                                }
                            )
                            tot += 1

        self.tot += tot
        print(f'ref-avs {self.test_name} sample nums: {tot}')


    def __len__(self):
        return len(self.samples)



    def __getitem__(self,idx):
        sample = self.samples[idx]
        uid = sample['uid']
        ref = sample.get('ref', sample.get('exp', ''))
        task_name = sample['task_name']
        instruction = sample['instruction']
        output = sample.get('output')
        
        data = {
            'uid':uid,
            'ref':ref,
            'instruction': "<s>" +instruction,
            'output':output + "</s>",
            'task_name':task_name,
        }
        
        if task_name == 'ref-avs':
            def _load_image(path: str):
                if path is None or not exists(path):
                    return None
                image = Image.open(path).convert('RGB')
                image = image.resize((224,224))
                image = self.video_processor.preprocess([image],return_tensors='pt')
                return image['pixel_values']

            def _load_video(image_paths, max_frames=None):
                if image_paths is None:
                    return None
                frames = []
                limit = len(image_paths) if max_frames is None else max_frames
                for path in image_paths[:limit]:
                    image = _load_image(path)
                    if image is not None:
                        frames.append(image)
                if len(frames) == 0:
                    return None
                return torch.cat(frames, dim=0)

            def _load_audio(audio_path: str):
                if audio_path is None or not exists(audio_path):
                    return None
                audio_feature = []
                audio, sr = librosa.load(audio_path,sr=16000,mono=True)
                length = len(audio)
                tot = 10
                nums_per_second = int(length / tot) if tot > 0 else 0
                indices = [i for i in range(tot)]
                for indice in indices:
                    start_time = max(0, indice)
                    end_time = min(tot, indice + 1)
                    audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                    if len(audio_seg) < 1 * nums_per_second:
                        sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                        audio_seg = np.concatenate((audio_seg, sil),axis=0)
                    audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                    fbank = preprocess(audio_seg)
                    fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                    audio_feature.append(fbank)
                if len(audio_feature) == 0:
                    return None
                return torch.stack(audio_feature,dim=0) # t,L,128

            # video context: image / video 模式都使用 10 帧上下文
            image_path_list = sample.get('image_path_list', []) or []
            # 如果样本里没给到完整帧列表，则默认从 0~9 构建
            if len(image_path_list) < self.video_frame_nums and sample.get('vid'):
                vid = sample['vid']
                fallback_list = [
                    join(getattr(self.data_args, "refavs_data_root", "../MQ-RAVSBench"),
                         'media', vid, 'frames', f"{i}.jpg")
                    for i in range(self.video_frame_nums)
                ]
                image_path_list = fallback_list
            frame_limit = self.video_frame_nums
            video = _load_video(image_path_list, max_frames=frame_limit)
            if video is not None:
                data['video'] = video
                data['video_path'] = image_path_list
                data['vid'] = sample.get('vid')

            image_path = sample.get('image_path')
            image = _load_image(image_path)
            if image is not None:
                data['image'] = image
                data['image_path'] = image_path

            audio_path = sample.get('audio_path')
            audio_feature = _load_audio(audio_path)
            if audio_feature is not None:
                data['audio'] = audio_feature
                data['audio_path'] = audio_path

            data['mask_path'] = sample.get('mask_path')
            data['mask_type'] = sample.get('mask_type')
            data['iou_to_gt'] = sample.get('iou_to_gt')
            data['action'] = sample.get('action')
            data['exp'] = sample.get('exp')
            data['frame_idx'] = sample.get('frame_idx')
            data['kfid'] = sample.get('kfid')

        return data



@dataclass
class DataCollatorForUnifiedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_processor: CLIPImageProcessor = None
    data_args: any = None

    def _build_mask_feature(self, mask_path: str, frame_path: str = None, mode: str = "mask"):
        """
        根据配置生成送入视觉编码器的 <mask> 特征。
        - mask: 仅输入黑白二值 mask。
        - masked_frame: 用 mask 掩码原始帧后再编码。
        - mask_and_masked_frame: 同时提供二者，沿时间维度拼接，便于模型做消融。
        - both: 兼容旧配置，随机在 mask / masked_frame 中二选一。
        """
        if self.image_processor is None:
            return None
        if mask_path is None or not exists(mask_path):
            return None

        def _load_mask_only():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None
            mask = (mask > 0).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask).convert('RGB')
            mask_img = mask_img.resize((224, 224))
            return self.image_processor.preprocess([mask_img], return_tensors='pt')['pixel_values']

        def _load_masked_frame():
            if frame_path is None or not exists(frame_path):
                return None
            frame = cv2.imread(frame_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if frame is None or mask is None:
                return None
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame * mask[..., None]
            masked_img = Image.fromarray(frame.astype(np.uint8)).convert('RGB')
            masked_img = masked_img.resize((224, 224))
            return self.image_processor.preprocess([masked_img], return_tensors='pt')['pixel_values']

        if mode in ['mask', 'binary_mask']:
            return _load_mask_only()
        if mode in ['masked_frame', 'masked_image']:
            return _load_masked_frame()
        if mode in ['mask_and_masked_frame', 'mask+masked_frame']:
            tensors = []
            for tensor in (_load_mask_only(), _load_masked_frame()):
                if tensor is not None:
                    tensors.append(tensor)
            if len(tensors) == 0:
                return None
            return torch.cat(tensors, dim=0)
        # 兼容旧的 both 表示随机二选一
        if mode == 'both':
            return _load_mask_only() if random.random() < 0.5 else _load_masked_frame()
        return None

    def __call__(self, instances: Sequence[Dict]): # 选择的 bs 个 data (__getitem__返回的)
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_task_names = []
        batch_iou_values = []
        batch_iou_indices = []

        refavs_pos_ratio = getattr(self.data_args, "refavs_pos_ratio", 0.5) if self.data_args is not None else 0.5
        mask_encode_mode = getattr(self.data_args, "refavs_mask_encode_mode", "masked_frame") if self.data_args is not None else "masked_frame"

        # 先抽取 ref-avs 样本，平衡正负 mask 选择
        ref_indices = [idx for idx, inst in enumerate(instances) if inst.get('task_name') == 'ref-avs']
        ref_pos_quota = int(len(ref_indices) * refavs_pos_ratio + 0.5) if len(ref_indices) > 0 else 0
        ref_pos_selected = 0
        selected_candidates = {}
        for idx in ref_indices:
            cand_list = instances[idx].get('mask_candidates', [])
            pos_list = [c for c in cand_list if c.get('mask_type') == 'perfect']
            neg_list = [c for c in cand_list if c.get('mask_type') != 'perfect']
            candidate = None
            if ref_pos_selected < ref_pos_quota and len(pos_list) > 0:
                candidate = random.choice(pos_list)
                ref_pos_selected += 1
            elif len(neg_list) > 0:
                type2cands = {}
                for c in neg_list:
                    type2cands.setdefault(c.get('mask_type', 'neg'), []).append(c)
                candidate = random.choice(type2cands[random.choice(list(type2cands.keys()))])
            elif len(pos_list) > 0:
                candidate = random.choice(pos_list)
            elif len(cand_list) > 0:
                candidate = random.choice(cand_list)
            selected_candidates[idx] = candidate

        iou_token_id = tokenizer.convert_tokens_to_ids("<iou_token>") if "<iou_token>" in tokenizer.get_vocab() else None


        for inst_idx, instance in enumerate(instances):
            instruction=instance['instruction']
            output=instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)

            # ref-avs：根据挑选的 mask 候选补充 output、mask 特征与 iou 监督
            if task_name == 'ref-avs':
                candidate = selected_candidates.get(inst_idx, None) # candidate 就是 add_ref_avs_samples 里 mask_candiates 的一项
                if candidate is not None:
                    output = candidate.get('response', '')
                    # response 字符串内已经包含 <think>/<answer>/<audit> 等标签，这里直接使用
                    if not output.endswith("</s>"):
                        output = output + "</s>"
                    iou_gt = round(float(candidate.get('iou_to_gt', 0.0)), 4)
                    instance['selected_iou'] = iou_gt
                    instance['selected_mask_type'] = candidate.get('mask_type', '')
                    # mask 模式选择
                    encode_mode = mask_encode_mode
                    # both: 随机二选一；mask_and_masked_frame: 同时拼接两种特征
                    if encode_mode == "both":
                        encode_mode = random.choice(["mask", "masked_frame"])
                    mask_feat = self._build_mask_feature(
                        candidate.get('mask_path'),
                        frame_path=instance.get('key_frame_path'),
                        mode=encode_mode,
                    )
                    if mask_feat is not None:
                        instance['mask'] = mask_feat
                else:
                    instance['selected_iou'] = None
                    instance['selected_mask_type'] = ''
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            input_ids = instruction_ids + output_ids
            label = [-100] * len(instruction_ids) + output_ids
            # print('instruction_ids', instruction_ids)
            # print('output_ids', output_ids)
            # print('label', label)
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))

            if task_name == 'ref-avs' and iou_token_id is not None:
                indices = torch.where(torch.tensor(input_ids, dtype=torch.long) == iou_token_id)[0]
                batch_iou_indices.append(indices)
                iou_value = instance.get('selected_iou', None)
                if iou_value is None:
                    iou_value = -1.0
                batch_iou_values.append(torch.tensor(iou_value, dtype=torch.float32))
            else:
                batch_iou_indices.append(torch.tensor([], dtype=torch.long))
                batch_iou_values.append(torch.tensor(-1.0, dtype=torch.float32))

            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image # key frame 特征
                
            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio
            
            mask = instance.get('mask',None) # 提供的掩码/掩码后的图像 特征
            if mask is not None:
                X_modals['<mask>'] = mask
            
            batch_X_modals.append(X_modals)

        if len(batch_iou_values) > 0:
            batch_iou_values = torch.stack(batch_iou_values, dim=0)
        else:
            batch_iou_values = torch.tensor([])

        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_task_names':batch_task_names,
            'batch_iou_values':batch_iou_values,
            'batch_iou_indices':batch_iou_indices,
        }


@dataclass
class DataCollatorForUnifiedTestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_processor: CLIPImageProcessor = None
    data_args: any = None

    def _build_mask_feature(self, mask_path: str, frame_path: str = None, mode: str = "mask"):
        """测试时根据配置构造 mask 特征，保持与训练阶段一致。"""
        if self.image_processor is None:
            return None
        if mask_path is None or not exists(mask_path):
            return None

        def _load_mask_only():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None
            mask = (mask > 0).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask).convert('RGB')
            mask_img = mask_img.resize((224, 224))
            return self.image_processor.preprocess([mask_img], return_tensors='pt')['pixel_values']

        def _load_masked_frame():
            if frame_path is None or not exists(frame_path):
                return None
            frame = cv2.imread(frame_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if frame is None or mask is None:
                return None
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame * mask[..., None]
            masked_img = Image.fromarray(frame.astype(np.uint8)).convert('RGB')
            masked_img = masked_img.resize((224, 224))
            return self.image_processor.preprocess([masked_img], return_tensors='pt')['pixel_values']

        if mode in ['mask', 'binary_mask']:
            return _load_mask_only()
        if mode in ['masked_frame', 'masked_image']:
            return _load_masked_frame()
        if mode in ['mask_and_masked_frame', 'mask+masked_frame']:
            tensors = []
            for tensor in (_load_mask_only(), _load_masked_frame()):
                if tensor is not None:
                    tensors.append(tensor)
            if len(tensors) == 0:
                return None
            return torch.cat(tensors, dim=0)
        if mode == 'both':
            return _load_mask_only() if random.random() < 0.5 else _load_masked_frame()
        return None

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_metadata=[] # 比 train多了这个
        batch_task_names = []

        for instance in instances:
            instruction = instance['instruction']
            output = instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)

            uid = instance.get('uid',None)
            ref = instance.get('ref',None)
            mask_type = instance.get('mask_type', None)
            iou_to_gt = instance.get('iou_to_gt', None)
            action = instance.get('action', None)
            frame_idx = instance.get('frame_idx', None)
            kfid = instance.get('kfid', None)
            vid = instance.get('vid', None)

            metadata = {
                'uid':uid,
                'ref':ref,
                'instruction': instruction,
                'output': output,
                'mask_type': mask_type,
                'iou_to_gt': iou_to_gt,
                'action': action,
                'frame_idx': frame_idx,
                'kfid': kfid,
                'vid': vid,
            }

            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            
            input_ids = instruction_ids
            label = [-100] * len(instruction_ids)
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image
                metadata['image_path'] = instance.get('image_path','')
                
            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video
                metadata['video_path'] = instance.get('video_path','')

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio
                metadata['audio_path'] = instance.get('audio_path','')
            
            mask = instance.get('mask',None)
            if mask is not None:
                X_modals['<mask>'] = mask
                metadata['mask_path'] = instance.get('mask_path','')
            else:
                # ref-avs 评测时根据路径动态生成 mask 特征
                mask_path = instance.get('mask_path', None)
                if mask_path is not None:
                    encode_mode = getattr(self.data_args, "refavs_mask_encode_mode", "masked_frame") if self.data_args is not None else "mask"
                    # both 依旧随机，mask_and_masked_frame 则同时提供两种输入
                    if encode_mode == "both":
                        encode_mode = random.choice(["mask", "masked_frame"])
                    mask_feature = self._build_mask_feature(mask_path, frame_path=instance.get('image_path'), mode=encode_mode)
                    if mask_feature is not None:
                        X_modals['<mask>'] = mask_feature
                        metadata['mask_path'] = mask_path

            # print('*' * 20)
            # print(metadata)
            # print('*' * 20)
            
            
            batch_X_modals.append(X_modals)
            batch_metadata.append(metadata)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_metadata':batch_metadata,
            'batch_task_names':batch_task_names,
        }


def get_dataset_collator(
    data_args,tokenizer: transformers.PreTrainedTokenizer,
    image_processor=None,mode='train',
    test_name = 'test_s',
):
    if mode == 'train':
        dataset = UnifiedDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            data_args=data_args,
            ref_avs_task=data_args.ref_avs_task,
            multi_frames=data_args.multi_frames,
        )
        data_collator = DataCollatorForUnifiedDataset(
            tokenizer=tokenizer,
            image_processor=image_processor,
            data_args=data_args,
        )

    elif mode == 'test':
        dataset = UnifiedTestDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            data_args=data_args,
            ref_avs_task=data_args.ref_avs_task,
            test_name=test_name,
            multi_frames=data_args.multi_frames,
        )
        data_collator = DataCollatorForUnifiedTestDataset(
            tokenizer=tokenizer,
            image_processor=image_processor,
            data_args=data_args,
        )
    
    return dataset,data_collator
