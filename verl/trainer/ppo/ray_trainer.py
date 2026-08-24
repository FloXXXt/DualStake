# ray_trainer-try2.py
#修改：参数温度等的传递；ece从main中导入
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, Optional

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
from verl.trainer.conf_adv_config import (
    LSC_MODE,
    FC_MODE,
    compute_ece,
    compute_auroc,
    apply_dual_calib_to_scores,
)

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses'] 
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),

        # metrics for actions
         'env/number_of_actions/mean':
            float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean()),
        'env/number_of_actions/max':
            float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max()),
        'env/number_of_actions/min':
            float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min()),
        'env/finish_ratio':
            1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean()),
        'env/number_of_valid_action':
            float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean()),
        'env/ratio_of_valid_action':
            float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean()),
        'env/number_of_valid_search':
            float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean()),
    }

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        import types
        reward_tensor_lst = []
        data_source_lst = []

        # ========== 跨 batch 累积统计 ==========
        # 所有字段均为 list，跨 batch 直接 extend
        all_reward_stats = None
        all_gen_stats = None

        def _merge_stats(accum, new_stats):
            """将 new_stats 各字段 extend 进 accum，返回合并后的 accum。"""
            if accum is None:
                # 第一批：深拷贝
                import copy
                return copy.deepcopy(new_stats)
            for k, v in new_stats.items():
                if isinstance(v, list) and k in accum and isinstance(accum[k], list):
                    accum[k].extend(v)
                # 非 list 字段（如不存在）忽略
            return accum
        # ==========================================

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                # ========== 跨 batch 累积统计（不在 loop 内计算，仅 extend）==========
                all_reward_stats = _merge_stats(all_reward_stats, test_batch.meta_info.get('reward_statistics', {}))
                all_gen_stats = _merge_stats(all_gen_stats, test_batch.meta_info.get('generation_statistics', {}))
                # ========================================================

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        ) 
                    
                    test_batch = test_batch.union(final_gen_batch_output)

                    # TODO-10 (val): transfer per_step_confidences from meta_info to non_tensor_batch
                    if 'per_step_confidences' in final_gen_batch_output.meta_info:
                        psc = final_gen_batch_output.meta_info['per_step_confidences']
                        test_batch.non_tensor_batch['per_step_confidences'] = np.array(psc, dtype=object)

                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()

                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    # ========== 跨 batch 累积统计（不在 loop 内计算，仅 extend）==========
                    all_reward_stats = _merge_stats(all_reward_stats, test_batch.meta_info.get('reward_statistics', {}))
                    all_gen_stats = _merge_stats(all_gen_stats, test_batch.meta_info.get('generation_statistics', {}))
                    # ========================================================

                    # ========== 每个 batch 立即保存样本级 CSV / JSONL ==========
                    _batch_gen_stats = test_batch.meta_info.get('generation_statistics', None)
                    _batch_reward_stats = test_batch.meta_info.get('reward_statistics', None)
                    if _batch_gen_stats and _batch_reward_stats:
                        self._save_sample_level_data(test_batch, _batch_gen_stats, _batch_reward_stats, split='val')
                    # ===========================================================

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        # ========== 全量统计：将所有 batch 累积后的 stats 一次性计算 ==========
        # 构造一个轻量 mock 对象，仅含 meta_info，传给 _collect_and_log_statistics
        if all_reward_stats or all_gen_stats:
            meta = {}
            if all_reward_stats:
                meta['reward_statistics'] = all_reward_stats
            if all_gen_stats:
                meta['generation_statistics'] = all_gen_stats
            mock_batch = types.SimpleNamespace(meta_info=meta)
            val_metrics = {}
            self._collect_and_log_statistics(mock_batch, val_metrics, split='val')
            if val_metrics:
                metric_dict.update(val_metrics)
        # ========== 全量统计结束 ==========

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator in ['grpo', 'reinforce']:
            self.use_critic = False
        else:
            self.use_critic = False

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()

                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                        # TODO-10 (train): transfer per_step_confidences from meta_info to non_tensor_batch
                        if 'per_step_confidences' in final_gen_batch_output.meta_info:
                            psc = final_gen_batch_output.meta_info['per_step_confidences']
                            batch.non_tensor_batch['per_step_confidences'] = np.array(psc, dtype=object)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    batch.meta_info['global_step'] = self.global_steps

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key not in ('old_log_probs', 'token_level_scores'):
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # ========== 双路 Calib Reward 叠加（lsc + fc） ==========
                        # 在 token_level_scores 写入之后、KL penalty 之前
                        # 从 token_level_scores 还原出 sequence-level outcome score（q），
                        # 叠加 lsc/fc calib 信号后写回（仍以 sequence-level scalar 放在最后一个有效位）
                        # batch['confidences']         = final-confidence（fc），用于评估 ECE/AUROC/CSV
                        # batch['last_step_confidences'] = lsc，用于 lsc 监督
                        if (LSC_MODE != "none" or FC_MODE != "none") and 'last_step_confidences' in batch.batch:
                            cur_step = batch.meta_info.get('global_step', 0)
                            uid_list = batch.non_tensor_batch['uid']
                            uid_list = uid_list.tolist() if hasattr(uid_list, 'tolist') else list(uid_list)

                            # 取出 sequence-level score：token_level_scores 在最后有效位有非零值
                            non_zero = (batch.batch['token_level_scores'] != 0)
                            seq_scores = (batch.batch['token_level_scores'] * non_zero).sum(dim=-1)  # (bs,)

                            # 双路叠加（返回修改后的 seq_scores）
                            new_seq_scores = apply_dual_calib_to_scores(
                                scores=seq_scores,
                                lsc_confs=batch.batch['last_step_confidences'],
                                fc_confs=batch.batch['confidences'],
                                uid_list=uid_list,
                                cur_step=cur_step,
                            )

                            # 写回 token_level_scores（只更新有非零值的位置）
                            delta = new_seq_scores - seq_scores  # (bs,)
                            last_nonzero_col = non_zero.long().flip(dims=[1]).argmax(dim=1)
                            last_nonzero_col = batch.batch['token_level_scores'].shape[1] - 1 - last_nonzero_col
                            for i in range(batch.batch['token_level_scores'].shape[0]):
                                batch.batch['token_level_scores'][i, last_nonzero_col[i]] += delta[i]
                        # =======================================================
                        
                        # 合并和记录统计信息
                        self._collect_and_log_statistics(batch, metrics, split='train')


                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n_agent)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
    # ========== 新增方法开始 ==========

    def _save_sample_level_data(self, batch: DataProto, gen_stats: dict, reward_stats: dict, split='train'):
        """
        保存每个样本的详细数据到 CSV 文件
        
        Args:
            batch: DataProto 对象
            gen_stats: generation 统计
            reward_stats: reward 统计
        """
        import csv
        import os

        uid = batch.non_tensor_batch.get("uid", None)
        if uid is None:
            uid = batch.non_tensor_batch.get("index", None)

        # normalize uid into a python list for indexing
        if isinstance(uid, np.ndarray):
            uid = uid.tolist()
        elif isinstance(uid, (list, tuple)):
            uid = list(uid)
        else:
            uid = None
        # ======== [FIX END] ========
        
        
        # 创建输出目录
        output_dir = os.path.join(self.config.trainer.default_local_dir, 'sample_statistics')
        os.makedirs(output_dir, exist_ok=True)
        
        # 🔴 根据配置自动生成文件名
        if split == 'val' and hasattr(self.config.trainer, 'csv_filename') and self.config.trainer.csv_filename:
            # 用户手动指定了文件名
            csv_filename = self.config.trainer.csv_filename
        elif split == 'val':
            # 自动生成验证集文件名
            # 1. 从 actor model path 提取信息
            model_path = self.config.actor_rollout_ref.model.path
            # 提取最后两级目录作为模型标识 (例如: actor/global_step_100)
            model_parts = model_path.rstrip('/').split('/')[-2:]
            model_id = '_'.join(model_parts)  # 例如: actor_global_step_100
            
            # 2. 从验证数据集路径提取数据集名称
            val_files = self.config.data.val_files
            if isinstance(val_files, (list, tuple)):
                val_files = val_files[0]
            # 提取数据集目录名 (例如: nq_hotpot-only)
            dataset_name = os.path.basename(os.path.dirname(val_files))
            
            # 3. 组合文件名
            csv_filename = f'{model_id}_{dataset_name}_val.csv'
        else:
            # 训练模式使用默认文件名
            csv_filename = 'sample_level_data.csv'
        
        csv_path = os.path.join(output_dir, csv_filename)
        
        # 检查文件是否存在,决定是否写入表头
        file_exists = os.path.exists(csv_path)
        
        # 打开文件并追加数据
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = [
                'global_step',
                'split',
                'sample_id',
                'data_source',
                'is_correct',
                'score',
                'total_output_tokens',
                'reasoning_tokens',
                'information_tokens',
                'search_count',
                'turn_count',
                'reasoning_ratio',  # reasoning_tokens / (reasoning_tokens + information_tokens)
                'has_valid_format',
                'answer_tag_count',
                'predicted_answer',
                'ground_truth',
                'think_open', 'think_close', 'think_malformed', 'think_total', 'think_bad',
                'search_open', 'search_close', 'search_malformed', 'search_total', 'search_bad',
                'answer_open', 'answer_close', 'answer_malformed', 'answer_total', 'answer_bad',
                # optional: all tags json (if you saved)
                'all_tag_open_json', 'all_tag_close_json', 'all_tag_malformed_json',
                'confidence',
                'confidence_valid',
                'confidence_reward',
                'is_overconfident',
                'is_underconfident',
                # per-step confidence (TODO-11)
                'per_step_conf_count',
                'per_step_conf_mean',
                'per_step_conf_values',
                'per_step_action_types',
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # 如果是新文件,写入表头
            if not file_exists:
                writer.writeheader()
            
            # 写入每个样本的数据
            num_samples = len(reward_stats['is_correct'])
            for i in range(num_samples):
                total_tokens = gen_stats['total_output_tokens'][i]
                reasoning_tokens = gen_stats['reasoning_tokens'][i]
                info_tokens = gen_stats['information_tokens'][i]
                
                # 计算 reasoning ratio
                if reasoning_tokens + info_tokens > 0:
                    reasoning_ratio = reasoning_tokens / (reasoning_tokens + info_tokens)
                else:
                    reasoning_ratio = 0.0
                
                row = {
                    'global_step': self.global_steps,
                    'split': split,
                    'sample_id': str(uid[i]) if (uid is not None and i < len(uid)) else str(i),
                    'data_source': reward_stats['data_sources'][i],
                    'is_correct': int(reward_stats['is_correct'][i]),
                    'score': reward_stats['scores'][i],
                    'total_output_tokens': total_tokens,
                    'reasoning_tokens': reasoning_tokens,
                    'information_tokens': info_tokens,
                    'search_count': gen_stats['search_count'][i],
                    'turn_count': gen_stats['turn_count'][i],
                    'reasoning_ratio': reasoning_ratio,
                    'has_valid_format': int(reward_stats['has_valid_format'][i]),
                    'answer_tag_count': reward_stats['answer_tag_count'][i],
                    'predicted_answer': reward_stats['predicted_answers'][i][:200],  # 截断
                    'ground_truth': reward_stats['ground_truths'][i][:200],
                    'think_open': int(reward_stats.get('think_open', [0]*num_samples)[i]),
                    'think_close': int(reward_stats.get('think_close', [0]*num_samples)[i]),
                    'think_malformed': int(reward_stats.get('think_malformed', [0]*num_samples)[i]),
                    'think_total': int(reward_stats.get('think_total', [0]*num_samples)[i]),
                    'think_bad': int(reward_stats.get('think_bad', [0]*num_samples)[i]),

                    'search_open': int(reward_stats.get('search_open', [0]*num_samples)[i]),
                    'search_close': int(reward_stats.get('search_close', [0]*num_samples)[i]),
                    'search_malformed': int(reward_stats.get('search_malformed', [0]*num_samples)[i]),
                    'search_total': int(reward_stats.get('search_total', [0]*num_samples)[i]),
                    'search_bad': int(reward_stats.get('search_bad', [0]*num_samples)[i]),

                    'answer_open': int(reward_stats.get('answer_open', [0]*num_samples)[i]),
                    'answer_close': int(reward_stats.get('answer_close', [0]*num_samples)[i]),
                    'answer_malformed': int(reward_stats.get('answer_malformed', [0]*num_samples)[i]),
                    'answer_total': int(reward_stats.get('answer_total', [0]*num_samples)[i]),
                    'answer_bad': int(reward_stats.get('answer_bad', [0]*num_samples)[i]),

                    'all_tag_open_json': reward_stats.get('all_tag_open_json', ['{}']*num_samples)[i],
                    'all_tag_close_json': reward_stats.get('all_tag_close_json', ['{}']*num_samples)[i],
                    'all_tag_malformed_json': reward_stats.get('all_tag_malformed_json', ['{}']*num_samples)[i],
                    # ========== 新增-confidence相关==========
                    'confidence': reward_stats.get('confidences', [None]*num_samples)[i],
                    'confidence_valid': int(reward_stats.get('confidence_valid', [False]*num_samples)[i]),
                    'confidence_reward': reward_stats.get('confidence_reward', [0.0]*num_samples)[i],
                    'is_overconfident': int(reward_stats.get('is_overconfident', [False]*num_samples)[i]),
                    'is_underconfident': int(reward_stats.get('is_underconfident', [False]*num_samples)[i]),
                    # per-step confidence (TODO-11)
                    'per_step_conf_count': reward_stats.get('per_step_conf_count', [0]*num_samples)[i],
                    'per_step_conf_mean': reward_stats.get('per_step_conf_mean', [None]*num_samples)[i],
                    'per_step_conf_values': str([s.get('confidence') for s in reward_stats.get('per_step_confidences', [[]] * num_samples)[i]]),
                    'per_step_action_types': str([s.get('action_type') for s in reward_stats.get('per_step_confidences', [[]] * num_samples)[i]]),
                    # ==========================

                }
                
                writer.writerow(row)
        
        # 每100步打印一次保存信息
        if self.global_steps % 100 == 0:
            print(f"[INFO] Saved {num_samples} {split} samples to {csv_path}")

        # ========== TODO-13: 保存 rollout 轨迹到 JSONL ==========
        import json

        # 与 CSV 文件命名对齐，前缀换为 trajectory_，扩展名换为 .jsonl
        if csv_filename == 'sample_level_data.csv':
            traj_filename = 'trajectory_train.jsonl'
        else:
            # val 模式：{model_id}_{dataset_name}_val.csv → trajectory_{model_id}_{dataset_name}_val.jsonl
            traj_filename = 'trajectory_' + csv_filename.replace('.csv', '.jsonl')

        traj_path = os.path.join(output_dir, traj_filename)

        # input_ids 已在 batch.batch 中，decode 还原完整轨迹文本（含 <information> 等 tag）
        input_ids_tensor = batch.batch.get('input_ids', None)
        # attention_mask 用于去掉 left-padding（Qwen 等模型 batch inference 时左填充）
        attention_mask_tensor = batch.batch.get('attention_mask', None)

        with open(traj_path, 'a', encoding='utf-8') as tf:
            for i in range(num_samples):
                traj_text = ''
                if input_ids_tensor is not None:
                    ids = input_ids_tensor[i]
                    # 用 attention_mask 截掉左侧 padding，只解码有效 token
                    if attention_mask_tensor is not None:
                        mask = attention_mask_tensor[i].bool()
                        ids = ids[mask]
                    traj_text = self.tokenizer.decode(
                        ids.tolist(),
                        skip_special_tokens=False
                    )
                record = {
                    'global_step': self.global_steps,
                    'split': split,
                    'sample_id': str(uid[i]) if (uid is not None and i < len(uid)) else str(i),
                    'data_source': reward_stats['data_sources'][i],
                    'is_correct': int(reward_stats['is_correct'][i]),
                    'score': reward_stats['scores'][i],
                    'trajectory': traj_text,
                }
                tf.write(json.dumps(record, ensure_ascii=False) + '\n')
        # ========== TODO-13 end ==========

    def _collect_and_log_statistics(self, batch, metrics: dict, split='train'):
        # 检查是否有统计信息（支持 DataProto 和 SimpleNamespace）
        gen_stats = batch.meta_info.get('generation_statistics', None)
        reward_stats = batch.meta_info.get('reward_statistics', None)

        has_gen_stats = gen_stats is not None and len(gen_stats) > 0
        has_reward_stats = reward_stats is not None and len(reward_stats) > 0

        if not (has_gen_stats or has_reward_stats):
            return

        print("\n" + "=" * 80)
        print("[COMBINED STATISTICS]")
        print("=" * 80)

        # 1) generation
        if has_gen_stats:
            metrics.update({
                'generation/avg_total_tokens': np.mean(gen_stats['total_output_tokens']),
                'generation/max_total_tokens': np.max(gen_stats['total_output_tokens']),
                'generation/min_total_tokens': np.min(gen_stats['total_output_tokens']),
                'generation/avg_reasoning_tokens': np.mean(gen_stats['reasoning_tokens']),
                'generation/avg_information_tokens': np.mean(gen_stats['information_tokens']),
                'generation/avg_search_count': np.mean(gen_stats['search_count']),
                'generation/max_search_count': np.max(gen_stats['search_count']),
                'generation/avg_turn_count': np.mean(gen_stats['turn_count']),
            })

            total_tokens = np.sum(gen_stats['total_output_tokens'])
            info_tokens = np.sum(gen_stats['information_tokens'])
            if total_tokens > 0:
                metrics['generation/info_token_ratio'] = info_tokens / total_tokens

            print(f"Generation Stats:")
            print(f"  Avg total tokens: {metrics['generation/avg_total_tokens']:.1f}")
            print(f"  Avg reasoning tokens: {metrics['generation/avg_reasoning_tokens']:.1f}")
            print(f"  Avg information tokens: {metrics['generation/avg_information_tokens']:.1f}")
            print(f"  Avg search count: {metrics['generation/avg_search_count']:.1f}")
            print(f"  Info token ratio: {metrics.get('generation/info_token_ratio', 0):.2%}")

        # 2) reward
        if has_reward_stats:
            total = len(reward_stats['is_correct'])
            correct = sum(reward_stats['is_correct'])
            accuracy = correct / total if total > 0 else 0

            metrics.update({
                'reward/accuracy': accuracy,
                'reward/correct_count': correct,
                'reward/total_count': total,
                'reward/avg_score': np.mean(reward_stats['scores']),
                'reward/valid_format_ratio': sum(reward_stats['has_valid_format']) / total if total > 0 else 0,
                'reward/avg_answer_tags': np.mean(reward_stats['answer_tag_count']),
            })

            # ========== 新增：confidence metrics ==========
            if has_reward_stats and 'confidences' in reward_stats:
                total = len(reward_stats['is_correct'])
                
                valid_confs = [c for c in reward_stats['confidences'] if c is not None]
                conf_valid_count = sum(reward_stats['confidence_valid'])
                overconf_count = sum(reward_stats['is_overconfident'])
                underconf_count = sum(reward_stats['is_underconfident'])

                metrics.update({
                    # 格式合规率
                    f'{split}/confidence/valid_rate':
                        conf_valid_count / total if total > 0 else 0,

                    # 平均confidence（只统计格式合规的）
                    f'{split}/confidence/mean':
                        float(np.mean(valid_confs)) if valid_confs else 0.0,

                    f'{split}/confidence/std':
                        float(np.std(valid_confs)) if valid_confs else 0.0,

                    # overconfidence / underconfidence比例
                    f'{split}/confidence/overconfident_rate':
                        overconf_count / total if total > 0 else 0,

                    f'{split}/confidence/underconfident_rate':
                        underconf_count / total if total > 0 else 0,

                    # confidence reward分量均值
                    f'{split}/confidence/avg_conf_reward':
                        float(np.mean(reward_stats['confidence_reward']))
                        if reward_stats['confidence_reward'] else 0.0,
                })

                # 答对/答错时的平均confidence（核心校准指标）
                correct_confs = [
                    reward_stats['confidences'][i]
                    for i in range(total)
                    if reward_stats['is_correct'][i]
                    and reward_stats['confidences'][i] is not None
                ]
                incorrect_confs = [
                    reward_stats['confidences'][i]
                    for i in range(total)
                    if not reward_stats['is_correct'][i]
                    and reward_stats['confidences'][i] is not None
                ]

                if correct_confs:
                    metrics[f'{split}/confidence/mean_when_correct'] = float(np.mean(correct_confs))
                if incorrect_confs:
                    metrics[f'{split}/confidence/mean_when_incorrect'] = float(np.mean(incorrect_confs))

                # 关键校准指标：答对时conf是否高于答错时conf
                # 这个gap越大说明calibration越好
                if correct_confs and incorrect_confs:
                    metrics[f'{split}/confidence/calibration_gap'] = (
                        float(np.mean(correct_confs)) - float(np.mean(incorrect_confs))
                    )

                # ECE近似（分桶计算）
                ece = compute_ece(reward_stats['confidences'], reward_stats['is_correct'])
                if ece is not None:
                    metrics[f'{split}/confidence/ece'] = ece
                    print(f"  ECE: {ece:.4f}  (越小越好，0=完美校准)")
                else:
                    print(f"  ECE: N/A (有效样本不足5个)"),
                
                auroc = compute_auroc(reward_stats['confidences'], reward_stats['is_correct'])
                if auroc is not None:
                    metrics[f'{split}/confidence/auroc'] = auroc
                    print(f"  AUROC: {auroc:.4f}  (越大越好，1=完美区分)")
                else:
                    print(f"  AUROC: N/A (有效样本不足或只有一类标签)")

                # ========== 当前奖励方案配置信息 ==========
                if split == 'train':
                    metrics['conf/lsc_mode'] = hash(LSC_MODE) % 1000  # 字符串转数字便于记录
                    metrics['conf/fc_mode']  = hash(FC_MODE)  % 1000
            # ================================================

            # ======== [FIX] tag totals only for TRAIN (wandb needs these) ========
            if split == "train":
                def _mean_list(key, default=0.0):
                    v = reward_stats.get(key, None)
                    if v is None or len(v) == 0:
                        return float(default)
                    return float(np.mean(v))

                metrics.update({
                    "train/tags/think_total_mean": _mean_list("think_total"),
                    "train/tags/search_total_mean": _mean_list("search_total"),
                    "train/tags/answer_total_mean": _mean_list("answer_total"),

                    "train/tags/think_bad_rate": _mean_list("think_bad"),
                    "train/tags/search_bad_rate": _mean_list("search_bad"),
                    "train/tags/answer_bad_rate": _mean_list("answer_bad"),

                    "train/tags/think_malformed_mean": _mean_list("think_malformed"),
                    "train/tags/search_malformed_mean": _mean_list("search_malformed"),
                    "train/tags/answer_malformed_mean": _mean_list("answer_malformed"),
                })
            # ======== [FIX END] ========

            print(f"\nReward Stats:")
            print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
            print(f"  Avg score: {metrics['reward/avg_score']:.3f}")
            print(f"  Valid format ratio: {metrics['reward/valid_format_ratio']:.2%}")

            # 按数据源统计
            ds_stats = {}
            for i, ds in enumerate(reward_stats['data_sources']):
                ds_stats.setdefault(ds, {'correct': 0, 'total': 0})
                ds_stats[ds]['total'] += 1
                if reward_stats['is_correct'][i]:
                    ds_stats[ds]['correct'] += 1

            print(f"\nAccuracy by data source:")
            for ds, st in ds_stats.items():
                ds_acc = st['correct'] / st['total'] if st['total'] > 0 else 0
                metrics[f'reward/accuracy_{ds}'] = ds_acc
                print(f"  {ds}: {ds_acc:.2%} ({st['correct']}/{st['total']})")

        # 3) combined + save csv
        if has_gen_stats and has_reward_stats:
            correct_indices = [i for i, c in enumerate(reward_stats['is_correct']) if c]
            incorrect_indices = [i for i, c in enumerate(reward_stats['is_correct']) if not c]

            if correct_indices:
                metrics['combined/correct_avg_tokens'] = np.mean([gen_stats['total_output_tokens'][i] for i in correct_indices])
                metrics['combined/correct_avg_searches'] = np.mean([gen_stats['search_count'][i] for i in correct_indices])
                metrics['combined/correct_avg_turns'] = np.mean([gen_stats['turn_count'][i] for i in correct_indices])

            if incorrect_indices:
                metrics['combined/incorrect_avg_tokens'] = np.mean([gen_stats['total_output_tokens'][i] for i in incorrect_indices])
                metrics['combined/incorrect_avg_searches'] = np.mean([gen_stats['search_count'][i] for i in incorrect_indices])
                metrics['combined/incorrect_avg_turns'] = np.mean([gen_stats['turn_count'][i] for i in incorrect_indices])

            # 只有真实 DataProto 对象（含 non_tensor_batch/batch 属性）才保存样本级数据
            # val 全量统计时传入的是 mock_batch，不包含 tensor 数据，跳过保存
            if hasattr(batch, 'non_tensor_batch') and hasattr(batch, 'batch'):
                self._save_sample_level_data(batch, gen_stats, reward_stats, split=split)

        print("=" * 80 + "\n")

    
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
