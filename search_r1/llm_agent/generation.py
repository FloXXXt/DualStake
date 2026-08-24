import torch
import re
import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor, is_final: bool = False) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at the appropriate tag.

        Normal turns (is_final=False):
          1. </search> found  → truncate at </search>  (per-step <confidence> is naturally included before)
          2. </final-confidence> found → truncate at </final-confidence>
          3. Neither → no truncation (invalid action, error prompt injected next turn)

        Final turn (is_final=True):
          1. </final-confidence> found → truncate at </final-confidence>
          2. Not found → no truncation (invalid action)
          Note: </search> is completely ignored in the final turn.
        """
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        new_responses_str = []
        for resp in responses_str:
            if not is_final:
                # Normal turn: prefer search, then final-confidence
                if '</search>' in resp:
                    new_responses_str.append(resp.split('</search>')[0] + '</search>')
                elif '</final-confidence>' in resp:
                    new_responses_str.append(resp.split('</final-confidence>')[0] + '</final-confidence>')
                else:
                    new_responses_str.append(resp)
            else:
                # Final turn: only truncate at final-confidence, ignore search
                if '</final-confidence>' in resp:
                    new_responses_str.append(resp.split('</final-confidence>')[0] + '</final-confidence>')
                else:
                    new_responses_str.append(resp)

        if self.config.no_think_rl:
            raise ValueError('stop')

        responses = self._batch_tokenize(new_responses_str)
        return responses, new_responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """Wrapper for generation that handles multi-GPU padding requirements."""
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def postprocess_predictions(self, predictions: List[Any], is_final: bool = False) -> Tuple[List[str], List[str], List[Any], List[Any]]:
        """Process predictions from llm into actions, contents, final_confidences and step_confidences.

        Returns:
            actions:           list of 'search' / 'answer' / None
            contents:          list of extracted content strings
            final_confidences: list of <final-confidence> values (float or None); only set when action='answer'
            step_confidences:  list of per-step <confidence> values (float or None);
                               only valid when the tag appears BEFORE the action tag
        """
        actions = []
        contents = []
        final_confidences = []
        step_confidences = []

        for prediction in predictions:
            if isinstance(prediction, str):
                # ------ 1. Extract action and content ------
                answer_match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
                if answer_match:
                    content = answer_match.group(1).strip()
                    action = 'answer'
                else:
                    search_match = re.search(r'<search>(.*?)</search>', prediction, re.DOTALL)
                    if search_match:
                        content = search_match.group(1).strip()
                        action = 'search'
                    else:
                        content = ''
                        action = None

                # ------ 2. Extract final-confidence (only when action='answer') ------
                final_conf = None
                if action == 'answer':
                    fc_match = re.search(r'<final-confidence>(.*?)</final-confidence>', prediction, re.DOTALL)
                    if fc_match:
                        try:
                            val = float(fc_match.group(1).strip())
                            if 0 <= val <= 10:
                                final_conf = val
                        except ValueError:
                            pass

                # ------ 3. Extract per-step confidence (position must be BEFORE the action tag) ------
                step_conf = None
                conf_match = re.search(r'<confidence>(.*?)</confidence>', prediction, re.DOTALL)
                if conf_match:
                    # Determine the start position of the action tag
                    action_tag_start = len(prediction)  # default: no action tag, no position limit
                    if action == 'search':
                        sm = re.search(r'<search>', prediction)
                        if sm:
                            action_tag_start = sm.start()
                    elif action == 'answer':
                        am = re.search(r'<answer>', prediction)
                        if am:
                            action_tag_start = am.start()
                    # Only accept if confidence tag ends before the action tag starts
                    if conf_match.end() <= action_tag_start:
                        try:
                            val = float(conf_match.group(1).strip())
                            if 0 <= val <= 10:
                                step_conf = val
                        except ValueError:
                            pass
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            actions.append(action)
            contents.append(content)
            final_confidences.append(final_conf)
            step_confidences.append(step_conf)

        return actions, contents, final_confidences, step_confidences

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True, is_final=False) -> Tuple[List[str], List[int], List[int], List[int], List[Any], List[Any]]:
        """Execute predictions and return observations, dones, validity, search flags, final_confidences, step_confidences."""
        cur_actions, contents, final_confidences, step_confidences = self.postprocess_predictions(predictions, is_final=is_final)
        next_obs, dones, valid_action, is_search = [], [], [], []

        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(
                        '\nMy previous action is invalid. '
                        'If I want to search, I should output <confidence>X</confidence> first (if I have received search results), '
                        'then <search> query </search>. '
                        'If I want to give the final answer, I should output <confidence>X</confidence> first (if I have received search results), '
                        'then <answer> my answer </answer>, then <final-confidence>X</final-confidence>. '
                        'Let me try again.\n'
                    )
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)

        assert len(search_results) == 0

        return next_obs, dones, valid_action, is_search, final_confidences, step_confidences

    def batch_search(self, queries: List[str] = None) -> str:
        """Batchified search for queries."""
        results = self._batch_search(queries)['result']
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop with statistics and confidence collection."""
        batch_size = gen_batch.batch['input_ids'].shape[0]

        # ========== Initialise statistics and confidence accumulators ==========
        generation_statistics = {
            'llm_output_tokens': [0] * batch_size,
            'information_tokens': [0] * batch_size,
            'total_output_tokens': [0] * batch_size,
            'reasoning_tokens': [0] * batch_size,
            'search_count': [0] * batch_size,
            'turn_count': [0] * batch_size,
        }

        # Final confidence (scalar per sample, written to batch['confidences'])
        batch_confidences_tensor = torch.full(
            (batch_size,), fill_value=-1, dtype=torch.float
        )
        # Per-step confidence: list of dicts per sample
        per_step_confidences = [[] for _ in range(batch_size)]
        # Whether each sample has received at least one <information> block
        has_retrieval = [False] * batch_size
        # ========== End of initialisation ==========

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # ==================== Main generation loop ====================
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'], is_final=False)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Token statistics for this turn
            current_response_tokens = (responses_ids != self.tokenizer.pad_token_id).sum(dim=1).tolist()
            for idx in range(batch_size):
                if active_mask[idx]:
                    generation_statistics['llm_output_tokens'][idx] += current_response_tokens[idx]
                    generation_statistics['turn_count'][idx] += 1

            # Execute predictions (now returns 6 values)
            next_obs, dones, valid_action, is_search, final_confs, step_confs = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, is_final=False
            )

            # Collect final confidence into tensor
            for idx in range(batch_size):
                if active_mask[idx] and final_confs[idx] is not None:
                    batch_confidences_tensor[idx] = final_confs[idx]

            # Collect per-step confidence record
            for idx in range(batch_size):
                if active_mask[idx]:
                    if is_search[idx]:
                        action_type = 'search'
                    elif dones[idx] and valid_action[idx]:
                        action_type = 'answer'
                    else:
                        action_type = 'invalid'
                    per_step_confidences[idx].append({
                        'turn': step + 1,               # 1-indexed
                        'confidence': step_confs[idx],  # None if missing or malformed
                        'had_information': has_retrieval[idx],
                        'action_type': action_type,
                    })

            # Update has_retrieval AFTER recording this step
            # (search result arrives in the NEXT turn as <information>)
            for idx in range(batch_size):
                if active_mask[idx] and is_search[idx]:
                    has_retrieval[idx] = True

            # Search count statistics
            for idx in range(batch_size):
                if is_search[idx]:
                    generation_statistics['search_count'][idx] += 1

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)

            # Information token statistics
            info_token_counts = (next_obs_ids != self.tokenizer.pad_token_id).sum(dim=1).tolist()
            for idx in range(batch_size):
                generation_statistics['information_tokens'][idx] += info_token_counts[idx]

            # Update rolling state and right side
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)

        # Ensure meta_info fields always exist (even if the loop runs zero times)
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()

        # ==================== Final rollout (with FINAL stage prompt) ====================
        if active_mask.sum():
            # ------ Inject FINAL stage prompt (rolling_state only, NOT right_side) ------
            final_stage_prompts = []
            for idx in range(batch_size):
                if active_mask[idx]:
                    if has_retrieval[idx]:
                        prompt = (
                            '\nThis is your final turn. You must provide an answer now. '
                            'Since you have received search results, first output your confidence in '
                            '<confidence> and </confidence>, then give your answer in <answer> and </answer>, '
                            'followed by your final confidence in <final-confidence> and </final-confidence>.\n'
                        )
                    else:
                        prompt = (
                            '\nThis is your final turn. You must provide an answer now. '
                            'Give your answer in <answer> and </answer>, followed by your final confidence '
                            'in <final-confidence> and </final-confidence>.\n'
                        )
                else:
                    prompt = ''  # placeholder for inactive samples
                final_stage_prompts.append(prompt)

            final_stage_ids = self._process_next_obs(final_stage_prompts)
            # Pass empty responses tensor (shape B×0) so _update_rolling_state only appends the stage prompt
            empty_responses = torch.zeros(
                (batch_size, 0), dtype=torch.long,
                device=rollings.batch['input_ids'].device
            )
            rollings = self._update_rolling_state(rollings, empty_responses, final_stage_ids)
            # NOTE: _update_right_side is intentionally NOT called here —
            #       the stage prompt must NOT be part of the response tokens / loss.

            # ------ Generate final response ------
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info.update(gen_output.meta_info)
            # Use is_final=True: only truncate at </final-confidence>, ignore </search>
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'], is_final=True)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Token statistics for final turn
            current_response_tokens = (responses_ids != self.tokenizer.pad_token_id).sum(dim=1).tolist()
            for idx in range(batch_size):
                if active_mask[idx]:
                    generation_statistics['llm_output_tokens'][idx] += current_response_tokens[idx]

            # Execute predictions for final turn
            _, dones, valid_action, is_search, final_confs, step_confs = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False, is_final=True
            )

            # Collect final confidence
            for idx in range(batch_size):
                if active_mask[idx] and final_confs[idx] is not None:
                    batch_confidences_tensor[idx] = final_confs[idx]

            # Collect per-step confidence for final turn (turn marked as "final")
            for idx in range(batch_size):
                if active_mask[idx]:
                    if is_search[idx]:
                        action_type = 'search'
                    elif dones[idx] and valid_action[idx]:
                        action_type = 'answer'
                    else:
                        action_type = 'invalid'
                    per_step_confidences[idx].append({
                        'turn': 'final',
                        'confidence': step_confs[idx],
                        'had_information': has_retrieval[idx],
                        'action_type': action_type,
                    })

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            meta_info['turns_stats'] = turns_stats.tolist()
            meta_info['active_mask'] = active_mask.tolist()
            meta_info['valid_action_stats'] = valid_action_stats.tolist()
            meta_info['valid_search_stats'] = valid_search_stats.tolist()

            original_right_side = self._update_right_side(original_right_side, responses_ids)
        
        # Compute reasoning tokens
        for idx in range(batch_size):
            generation_statistics['total_output_tokens'][idx] = (
                generation_statistics['llm_output_tokens'][idx] +
                generation_statistics['information_tokens'][idx]
            )
            generation_statistics['reasoning_tokens'][idx] = generation_statistics['llm_output_tokens'][idx]

        # ========== Write confidences and statistics into meta_info ==========
        original_right_side['confidences'] = batch_confidences_tensor

        # ---------- Build last_step_confidences: <confidence> tag at the answer step ----------
        # For each sample, find the last per_step_confidences entry with action_type=='answer'
        # and use its 'confidence' value. Fallback to -1 if not found or value is None.
        last_step_conf_tensor = torch.full(
            (batch_size,), fill_value=-1, dtype=torch.float
        )
        for idx in range(batch_size):
            for entry in reversed(per_step_confidences[idx]):
                if entry.get('action_type') == 'answer':
                    val = entry.get('confidence')
                    if val is not None:
                        last_step_conf_tensor[idx] = val
                    break  # found the answer step (with or without valid conf), stop
        original_right_side['last_step_confidences'] = last_step_conf_tensor

        meta_info['generation_statistics'] = generation_statistics
        meta_info['per_step_confidences'] = per_step_confidences  # TODO-8: pass per-step conf downstream

        # ---------- Print final confidence statistics ----------
        valid_confs = batch_confidences_tensor[batch_confidences_tensor != -1].tolist()
        print("\n" + "=" * 80)
        print("[CONFIDENCE STATISTICS]")
        print("=" * 80)
        print(f"Samples with valid final-confidence: {len(valid_confs)}/{batch_size}")
        if valid_confs:
            print(f"Average final-confidence: {np.mean(valid_confs):.2f}")
            print(f"Min: {min(valid_confs)}  Max: {max(valid_confs)}")

        # ---------- Print per-step confidence statistics (TODO-7) ----------
        all_step_conf_vals = [
            entry['confidence']
            for sample in per_step_confidences
            for entry in sample
            if entry['confidence'] is not None
        ]
        total_steps = sum(len(sample) for sample in per_step_confidences)
        steps_with_conf = len(all_step_conf_vals)
        search_with_conf = sum(
            1 for sample in per_step_confidences
            for entry in sample
            if entry['action_type'] == 'search' and entry['confidence'] is not None
        )
        answer_with_conf = sum(
            1 for sample in per_step_confidences
            for entry in sample
            if entry['action_type'] == 'answer' and entry['confidence'] is not None
        )
        print(f"\n[PER-STEP CONFIDENCE STATISTICS]")
        print(f"Total active steps: {total_steps}")
        print(f"Steps with valid per-step confidence: {steps_with_conf}/{total_steps}")
        if all_step_conf_vals:
            print(f"Avg per-step confidence: {np.mean(all_step_conf_vals):.2f}")
        print(f"Search steps with conf: {search_with_conf}  |  Answer steps with conf: {answer_with_conf}")

        # ---------- Print last_step_confidences statistics ----------
        valid_lsc = last_step_conf_tensor[last_step_conf_tensor != -1].tolist()
        print(f"\n[LAST-STEP CONFIDENCE STATISTICS] (used for stake reward)")
        print(f"Samples with valid last-step <confidence>: {len(valid_lsc)}/{batch_size}")
        if valid_lsc:
            print(f"Avg last-step confidence: {np.mean(valid_lsc):.2f}  Min: {min(valid_lsc)}  Max: {max(valid_lsc)}")

        # ---------- Print token statistics ----------
        print("\n[GENERATION STATISTICS]")
        print("-" * 80)
        print(f"Batch size: {batch_size}")
        print(f"Total output tokens: {np.mean(generation_statistics['total_output_tokens']):.1f}")
        print(f"  - LLM output tokens: {np.mean(generation_statistics['llm_output_tokens']):.1f}")
        print(f"  - Information tokens: {np.mean(generation_statistics['information_tokens']):.1f}")
        print(f"Avg search count: {np.mean(generation_statistics['search_count']):.1f}")
        print(f"Avg turns: {np.mean(generation_statistics['turn_count']):.1f}")
        print("=" * 80 + "\n")

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output