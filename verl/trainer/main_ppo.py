# main_ppo-try2.py
# 修改：confidence作为系数，具体实现包括softmax归一化处理和温度
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""


from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qa_f1
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

from verl.trainer.conf_adv_config import (
    compute_ece,
    compute_auroc,
    LSC_MODE,
    FC_MODE,
)

# ================== Reward 默认配置 ==================
REWARD_TYPE = "f1"
# reward_mode="base" 表示只用 r_answer + r_format + r_calib，去掉搜索/回合/无答案惩罚
REWARD_MODE = "base"

# 以下三项在新 reward_mode="base" 下不参与计算，保留参数供旧模式兼容
NO_SEARCH_PENALTY = 0.05
TURN_FREE = 1
TURN_PENALTY_PER_TURN = 0
NO_ANSWER_PENALTY = 0.05
CURRICULUM_STEP = 0

# confidence 统计（用于 logging，不影响 reward 计算）
# calib reward 由 ray_trainer.py 中 apply_dual_calib_to_scores 在组内计算
# 通过 conf_adv_config.py 的 LSC_MODE / FC_MODE 参数控制
CONFIDENCE_MODE = "observe_only"      # RewardManager 只做统计，不计算 r_calib
CONFIDENCE_MODE_VAL = "observe_only"  # 验证时同样只统计

# F1 >= CORRECT_THRESHOLD 算"答对"，用于 logging 中 is_correct 判定
CORRECT_THRESHOLD = 0.8

# ---- r_format 参数 ----
# <answer> 格式正确时给的小正向奖励
FORMAT_SCORE = 0.1


# ================== Per-step Confidence 配置 ==================
ENABLE_PERSTEP_REWARD = False          # 细粒度 per-step reward 开关（默认关闭）
PERSTEP_CONF_MISSING_PENALTY = 0.02   # per-step conf 缺失时的格式惩罚（粗粒度模式）


def _select_rm_score_fn(data_source, reward_type: str = "em"):
    """
    修改：新增正确性检测选择em/f1
    """
    if data_source.lower() in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', '2wiki', 'musique', 'bamboogle', 'nq_hotpotqa', 'nqhotpot', 'simpleqa']:
        if reward_type == "em":
            return qa_em.compute_score_em
        elif reward_type == "f1":
            return qa_f1.compute_score_f1
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
    else:
        raise NotImplementedError(f"Unknown data_source: '{data_source}'. Supported: ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', '2wiki', 'musique', 'bamboogle', 'nq_hotpotqa', 'nqhotpot', 'simpleqa']")


class RewardManager():
    """The reward manager.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        format_score: float = FORMAT_SCORE,
        reward_type: str = REWARD_TYPE,
        reward_mode: str = REWARD_MODE,
        no_search_penalty: float = NO_SEARCH_PENALTY,
        turn_free: int = TURN_FREE,
        turn_penalty_per_turn: float = TURN_PENALTY_PER_TURN,
        no_answer_penalty: float = NO_ANSWER_PENALTY,
        curriculum_step: int = CURRICULUM_STEP,
        # confidence 统计相关参数
        confidence_mode: str = CONFIDENCE_MODE,
        correct_threshold: float = CORRECT_THRESHOLD,
        # 评估模式：final_score 只用 EM（base_score），不叠加任何惩罚
        eval_only_em: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score

        self.reward_type = reward_type
        self.reward_mode = reward_mode
        self.no_search_penalty = float(no_search_penalty)
        self.turn_free = int(turn_free)
        self.turn_penalty_per_turn = float(turn_penalty_per_turn)
        self.no_answer_penalty = float(no_answer_penalty)
        self.curriculum_step = int(curriculum_step)

        self.confidence_mode = confidence_mode
        self.correct_threshold = float(correct_threshold)
        self.eval_only_em = eval_only_em

        print("=" * 80)
        print("[REWARD_MANAGER] Initialization:")
        print(f"  reward_type: {self.reward_type}")
        print(f"  reward_mode: {self.reward_mode}")
        print(f"  format_score: {self.format_score}")
        print(f"  num_examine: {self.num_examine}")
        print(f"  no_search_penalty: {self.no_search_penalty}")
        print(f"  turn_free: {self.turn_free}")
        print(f"  turn_penalty_per_turn: {self.turn_penalty_per_turn}")
        print(f"  no_answer_penalty: {self.no_answer_penalty}")
        print(f"  curriculum_step: {self.curriculum_step}")
        print(f"  confidence_mode: {self.confidence_mode}")
        print(f"  correct_threshold: {self.correct_threshold}")
        print(f"  [calib reward] LSC_MODE={LSC_MODE}, FC_MODE={FC_MODE}")
        print(f"  eval_only_em: {self.eval_only_em}")
        print("=" * 80)

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # ####debug打印确认confidence tensor信息####
        # # 验证 confidence tensor
        # if 'confidences' in data.batch:
        #     confidence_tensor = data.batch['confidences']
        #     print(f"[DEBUG] confidence tensor shape: {confidence_tensor.shape}, dtype: {confidence_tensor.dtype}")
        #     print(f"[DEBUG] confidence values: min={confidence_tensor.min():.2f}, max={confidence_tensor.max():.2f}")
        #     print(f"[DEBUG] invalid count (-1): {(confidence_tensor == -1).sum().item()}")
            
        #     # 验证顺序一致性（可选）
        #     # 比较第一个样本的 confidence 和 responses 内容是否对应
        #     if self.num_examine > 0:
        #         first_conf = confidence_tensor[0].item()
        #         first_response = self.tokenizer.decode(data.batch['responses'][0])
        #         print(f"[DEBUG] Sample 0: confidence={first_conf}, response contains <confidence>: {'<confidence>' in first_response}")
        # ###至此###

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        print(f"[DEBUG] Reward type in use: {self.reward_type}")

        # ========== 修改：从batch取confidence tensor而不是meta_info ==========
        # confidence存在batch里，shape=(batch_size,)，dtype=long
        # -1表示格式错误或未输出，1-10表示合法值
        if 'confidences' in data.batch:
            confidence_tensor = data.batch['confidences']
            has_confidence = True
        else:
            confidence_tensor = None
            has_confidence = False
            print("[DEBUG] No confidence tensor found in batch")
        # =====================================================================


        # ========== 新增 ==========
        reward_statistics = {
            'is_correct': [],
            'predicted_answers': [],
            'ground_truths': [],
            'scores': [],
            'data_sources': [],
            'answer_tag_count': [],
            'has_valid_format': [],
            'n_search': [],
            # ========== 新增：confidence统计 ==========
            'confidences': [],           # 原始confidence值（None表示格式错误）
            'confidence_valid': [],      # bool，confidence格式是否正确
            'confidence_reward': [],     # 每个样本的confidence reward分量
            'is_overconfident': [],      # bool，答错但conf>=7
            'is_underconfident': [],     # bool，答对但conf<=3
            # per-step confidence统计
            'per_step_confidences': [],      # 每个样本的 per_step_confidences list
            'per_step_conf_count': [],       # 每个样本有效 per-step conf 数量
            'per_step_conf_mean': [],        # 每个样本 per-step conf 均值（None 表示无有效值）
            # ==========================================
            
        }
        # ========== 新增结束 ==========

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # ===== decode: 利用info-mask 分开 decode，后续只用 response 做匹配 =====
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            # response token ids
            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            # ---- action-only filtering (key!) ----
            action_only_ids = valid_response_ids
            if 'info_mask' in data_item.batch:
                # info_mask 是整段 [prompt + response] 的 mask
                valid_info_mask = data_item.batch['info_mask'][prompt_length:prompt_length + valid_response_length]
                # 经验上：1 = action(token from model), 0 = obs/info
                action_only_ids = valid_response_ids[valid_info_mask.bool()]


            response_str = self.tokenizer.decode(action_only_ids)


            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            compute_score_fn = _select_rm_score_fn(data_source, reward_type=self.reward_type)

            # ===== base correctness reward: 只用 response_str =====
            base_score = compute_score_fn(
                solution_str=response_str,
                ground_truth=ground_truth,
                format_score=self.format_score
            )

            # ===== 取 global_step（用于 curriculum），ray_trainer 会补；拿不到就 -1 =====
            cur_step = -1
            if hasattr(data, "meta_info") and isinstance(data.meta_info, dict):
                cur_step = int(data.meta_info.get("global_step", -1))

            # ===== n_search（不搜惩罚用） =====
            n_search = 0
            if hasattr(data, "meta_info") and 'valid_search_stats' in data.meta_info:
                try:
                    n_search = int(data.meta_info['valid_search_stats'][i])
                except Exception:
                    n_search = 0

            # ===== turn_count（多回合惩罚用） =====
            turn_count = 0
            if hasattr(data, "meta_info") and "turns_stats" in data.meta_info:
                try:
                    turn_count = int(data.meta_info["turns_stats"][i])
                except Exception:
                    turn_count = 0


            # ===== answer 提取：只从 response_str 里提取 <answer> =====
            predicted_answer, answer_tag_count = self._extract_answer_with_format_check(response_str)

            # ===== GT 文本提取（你原来的安全转换逻辑保留）=====
            if isinstance(ground_truth, dict) and 'target' in ground_truth:
                raw_gt = ground_truth['target']
            else:
                raw_gt = ground_truth

            if isinstance(raw_gt, np.ndarray):
                gt_text = str(raw_gt.reshape(-1)[0]) if raw_gt.size > 0 else ""
            elif isinstance(raw_gt, (list, tuple)):
                gt_text = str(raw_gt[0]) if len(raw_gt) > 0 else ""
            else:
                gt_text = str(raw_gt)

            # ===== no_answer 判定（空/and/占位）=====
            is_no_answer = self._is_no_answer(predicted_answer)

            # ===== has_valid_format：response 里至少 1 个 answer tag 且 answer 非空非占位 =====
            has_valid_format = (answer_tag_count >= 1) and (not is_no_answer)

            # ===== 旧惩罚项（仅在兼容旧 reward_mode 时生效，新 "base" 模式下全为 0）=====
            no_search_pen = 0.0
            if self.reward_mode in {"seg_turn", "seg_turn_curr", "seg_turn_curr_ans"}:
                if n_search == 0:
                    no_search_pen = -self.no_search_penalty

            turn_pen = 0.0
            if self.reward_mode in {"seg_turn", "seg_turn_curr", "seg_turn_curr_ans"}:
                enable_turn_pen = True
                if self.reward_mode in {"seg_turn_curr", "seg_turn_curr_ans"}:
                    enable_turn_pen = (cur_step >= self.curriculum_step)
                if enable_turn_pen:
                    extra_turn = max(0, int(turn_count) - int(self.turn_free))
                    turn_pen = -self.turn_penalty_per_turn * extra_turn

            ans_pen = 0.0
            if self.reward_mode in {"ans_only", "seg_turn_curr_ans"}:
                if (answer_tag_count == 0) or is_no_answer:
                    ans_pen = -self.no_answer_penalty

            # ===== r_format：格式合规奖励 =====
            # 两部分：answer 格式正向奖励 + confidence 格式惩罚，独立叠加
            r_format = 0.0
            if has_valid_format:
                r_format += self.format_score          # <answer> 格式正确 → +0.1

            # ===== confidence 提取（只统计，不计算 r_calib）=====
            # r_calib 已移至 ray_trainer.py 的 apply_dual_calib_to_scores 按组计算
            confidence = None
            is_conf_valid = False
            is_overconfident = False
            is_underconfident = False

            if has_confidence and self.confidence_mode != "none":
                conf_val = confidence_tensor[i].item()
                confidence = conf_val if conf_val != -1 else None
                is_conf_valid = (confidence is not None)
                is_correct_for_conf = (float(base_score) >= self.correct_threshold)

                if is_conf_valid:
                    conf_normalized = confidence / 10.0
                    if not is_correct_for_conf and conf_normalized > 0.75:
                        is_overconfident = True
                    elif is_correct_for_conf and conf_normalized < 0.25:
                        is_underconfident = True


            # ===== per-step confidence reward =====
            psc_list = []
            if 'per_step_confidences' in data.non_tensor_batch:
                raw_psc = data.non_tensor_batch['per_step_confidences'][i]
                psc_list = list(raw_psc) if raw_psc is not None else []

            if not ENABLE_PERSTEP_REWARD:
                missing_steps = sum(
                    1 for s in psc_list
                    if s.get('had_information') and s.get('confidence') is None
                )
                if missing_steps > 0:
                    r_format += -PERSTEP_CONF_MISSING_PENALTY * missing_steps
            # else: fine-grained per-step reward（TODO: implement later）

            if self.eval_only_em:
                # 评估模式：只用纯 EM 得分
                final_score = float(base_score)
            else:
                # final_score = r_answer + r_format（calib 信号由 ray_trainer 在组内叠加）
                final_score = (
                    float(base_score)
                    + float(r_format)
                    + float(no_search_pen)   # 旧模式兼容，base 模式下恒 0
                    + float(turn_pen)        # 旧模式兼容，base 模式下恒 0
                    + float(ans_pen)         # 旧模式兼容，base 模式下恒 0
                )
            reward_tensor[i, valid_response_length - 1] = final_score

            # ===== 统计项（同步更新为 response-only 语义）=====
            if self.reward_type == "em":
                is_correct = bool(base_score == 1.0)
            else:  # f1
                is_correct = bool(base_score >= 0.8)


            reward_statistics['is_correct'].append(is_correct)
            reward_statistics['predicted_answers'].append(predicted_answer)
            reward_statistics['ground_truths'].append(gt_text)
            reward_statistics['scores'].append(float(final_score))
            reward_statistics['data_sources'].append(data_source)
            reward_statistics['answer_tag_count'].append(answer_tag_count)
            reward_statistics['has_valid_format'].append(has_valid_format)
            reward_statistics['n_search'].append(n_search)
            # ========== 新增：confidence统计 ==========
            reward_statistics['confidences'].append(confidence)
            reward_statistics['confidence_valid'].append(is_conf_valid)
            reward_statistics['confidence_reward'].append(0.0)  # calib由ray_trainer计算
            reward_statistics['is_overconfident'].append(is_overconfident)
            reward_statistics['is_underconfident'].append(is_underconfident)
            # per-step confidence
            reward_statistics['per_step_confidences'].append(psc_list)
            valid_psc_vals = [s['confidence'] for s in psc_list if s.get('confidence') is not None]
            reward_statistics['per_step_conf_count'].append(len(valid_psc_vals))
            reward_statistics['per_step_conf_mean'].append(
                float(np.mean(valid_psc_vals)) if valid_psc_vals else None
            )
            # ==========================================


            # ===== debug example（强制打印 penalty 分解 + response head）=====
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("\n" + "=" * 80)
                print(f"[REWARD EXAMPLE] Data Source: {data_source}")
                print("=" * 80)
                print(f"[step] cur_step={cur_step}")
                print(f"[mode] reward_type={self.reward_type} reward_mode={self.reward_mode} confidence_mode={self.confidence_mode}")
                print("-" * 80)
                print("[PROMPT head]")
                print(prompt_str[:300].replace("\n", "\\n"))
                print("-" * 80)
                print("[RESPONSE head]")
                print(response_str[:500].replace("\n", "\\n"))
                print("-" * 80)
                print(f"Predicted Answer: {predicted_answer!r}")
                print(f"Ground Truth: {gt_text!r}")
                print(f"Base Score: {base_score}")
                print(f"Confidence: {confidence} (valid={is_conf_valid})")
                print(f"is_overconfident={is_overconfident} is_underconfident={is_underconfident}")
                print(f"r_format={r_format:.4f}  [calib by ray_trainer, lsc_mode={LSC_MODE}, fc_mode={FC_MODE}]")
                if self.reward_mode != "base":
                    print(f"  [legacy] no_search={no_search_pen:.4f}, turn={turn_pen:.4f}, ans={ans_pen:.4f}")
                print(f"Final Score: {final_score:.4f}  (base={base_score:.4f})")
                print("=" * 80 + "\n")

        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        # ========== 新增 ==========
        data.meta_info['reward_statistics'] = reward_statistics
        
        # ========== 打印统计摘要（新增confidence部分）==========
        total = len(reward_statistics['is_correct'])
        correct = sum(reward_statistics['is_correct'])
        valid_format = sum(reward_statistics['has_valid_format'])

        print("\n" + "=" * 80)
        print("[REWARD STATISTICS]")
        print("=" * 80)
        print(f"Batch size: {total}")
        print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"Valid format: {valid_format}/{total} ({valid_format/total*100:.1f}%)")
        print(f"Avg score: {np.mean(reward_statistics['scores']):.3f}")

        # confidence统计摘要
        if has_confidence and self.confidence_mode != "none":
            valid_confs = [c for c in reward_statistics['confidences'] if c is not None]
            conf_valid_count = sum(reward_statistics['confidence_valid'])
            overconf_count = sum(reward_statistics['is_overconfident'])
            underconf_count = sum(reward_statistics['is_underconfident'])

            print(f"\nConfidence Stats (mode={self.confidence_mode}):")
            print(f"  Valid confidence: {conf_valid_count}/{total} ({conf_valid_count/total*100:.1f}%)")
            if valid_confs:
                print(f"  Avg confidence: {np.mean(valid_confs):.2f}")
                print(f"  Confidence distribution: {np.histogram(valid_confs, bins=range(1,12))[0].tolist()}")
            print(f"  Overconfident samples: {overconf_count}/{total} ({overconf_count/total*100:.1f}%)")
            print(f"  Underconfident samples: {underconf_count}/{total} ({underconf_count/total*100:.1f}%)")
            print(f"  Avg confidence reward: {np.mean(reward_statistics['confidence_reward']):.4f}")
            ece = compute_ece(
                reward_statistics['confidences'],
                reward_statistics['is_correct']
            )
            auroc = compute_auroc(
                reward_statistics['confidences'],
                reward_statistics['is_correct']
            )
            if ece is not None:
                print(f"  ECE: {ece:.4f}")
            else:
                print(f"  ECE: N/A")

            if auroc is not None:
                print(f"  AUROC: {auroc:.4f}  (越大越好，1=完美区分)")
            else:
                print(f"  AUROC: N/A (有效样本不足或只有一类标签)")

            # 分组分析：答对/答错的confidence分布
            correct_confs = [reward_statistics['confidences'][i]
                           for i in range(total)
                           if reward_statistics['is_correct'][i]
                           and reward_statistics['confidences'][i] is not None]
            incorrect_confs = [reward_statistics['confidences'][i]
                              for i in range(total)
                              if not reward_statistics['is_correct'][i]
                              and reward_statistics['confidences'][i] is not None]
            if correct_confs:
                print(f"  Avg confidence when correct: {np.mean(correct_confs):.2f}")
            if incorrect_confs:
                print(f"  Avg confidence when incorrect: {np.mean(incorrect_confs):.2f}")

        ds_stats = {}
        for i, ds in enumerate(reward_statistics['data_sources']):
            if ds not in ds_stats:
                ds_stats[ds] = {'correct': 0, 'total': 0}
            ds_stats[ds]['total'] += 1
            if reward_statistics['is_correct'][i]:
                ds_stats[ds]['correct'] += 1

        print("\nAccuracy by data source:")
        for ds, stats in ds_stats.items():
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {ds}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        print("=" * 80 + "\n")

        return reward_tensor
    
    # ========== 新增方法 ==========
    def _extract_answer_with_format_check(self, text: str) -> tuple:
        """只在 response 文本中提取 <answer>，返回(last_answer, tag_count)"""
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = list(re.finditer(answer_pattern, text or "", re.DOTALL))
        tag_count = len(matches)
        if tag_count == 0:
            return "", 0
        answer = matches[-1].group(1).strip()
        return answer, tag_count

    def _is_no_answer(self, ans: str) -> bool:
        """判定 answer 是否为空/占位（防止骗 answer 惩罚）"""
        if ans is None:
            return True
        x = str(ans).strip()
        if x == "":
            return True
        # 用和 F1 相同的 normalize（更稳）
        try:
            x_norm = qa_f1.normalize_answer(x)
        except Exception:
            x_norm = x.lower().strip()

        # 你提到的 and，以及一些常见占位
        if x_norm in {"and", "or", "the", "a", "an", "none", "null"}:
            return True

        # 太短且全是停用占位（可选更严格）
        toks = x_norm.split()
        if len(toks) <= 2 and all(t in {"and", "or", "the", "a", "an"} for t in toks):
            return True
        return False

    # ========== 新增结束 ==========


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id
    
    reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        reward_type=REWARD_TYPE,
        reward_mode=REWARD_MODE,
        no_search_penalty=NO_SEARCH_PENALTY,
        turn_free=TURN_FREE,
        turn_penalty_per_turn=TURN_PENALTY_PER_TURN,
        no_answer_penalty=NO_ANSWER_PENALTY,
        curriculum_step=CURRICULUM_STEP,
        confidence_mode=CONFIDENCE_MODE,
        correct_threshold=CORRECT_THRESHOLD,
    )

    # val 建议只做纯 correctness（便于比较）em得分，方便多项目比较，也可以保持同一 mode 看稳定性
    val_reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        reward_type='em',
        reward_mode="base",   # 推荐：验证只看 base correctness
        # observe_only：统计confidence但不影响EM分数
        confidence_mode=CONFIDENCE_MODE_VAL,
        correct_threshold=CORRECT_THRESHOLD,
        # 评估时只用纯 EM 得分，不叠加任何惩罚
        eval_only_em=True,
        # ====================================================
    )


    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__': 
    main()
