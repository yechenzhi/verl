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

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.debug import marked_timer

import threading
import json
import torch
import numpy as np
import uuid
from collections import defaultdict

class AsyncRolloutSaver:
    """
    一个异步将超长 rollout 保存到文件的类。
    它使用一个单独的线程来执行文件I/O，以避免阻塞主训练循环。
    数据被保存为 JSON Lines (.jsonl) 格式。
    """
    def __init__(self, filepath: str):
        """
        初始化 Saver。
        :param filepath: 保存数据的文件路径。
        """
        self.filepath = filepath
        self._lock = threading.Lock()
        print(f"异步 Rollout Saver 已初始化，将保存超长序列到: {self.filepath}")

    def _prepare_data_for_serialization(self, batch_data, indices, step: int):
        """
        从 DataProto 批次中提取指定索引的数据，并准备成可序列化的格式。
        """
        data_to_save = []
        
        # 从 non_tensor_batch 中获取 uids (如果存在)
        uids = batch_data.non_tensor_batch.get("uid", [None] * len(batch_data))
        
        for idx in indices:
            # 将张量转换为列表以便JSON序列化
            # 您可以根据需要保存更多或更少的信息
            rollout_info = {
                "step": step,
                "uid": uids[idx],
                "input_ids": batch_data.batch["input_ids"][idx].tolist(),
                "attention_mask": batch_data.batch["attention_mask"][idx].tolist(),
                "sequence_length": int(batch_data.batch["attention_mask"][idx].sum())
            }
            
            # 如果有其他想保存的张量或非张量数据，也可以在这里添加
            # 例如: token_level_rewards
            if "token_level_rewards" in batch_data.batch:
                rollout_info["token_level_rewards"] = batch_data.batch["token_level_rewards"][idx].tolist()

            data_to_save.append(rollout_info)
            
        return data_to_save

    def _save_worker(self, data_list: list):
        """
        在单独的线程中运行的工作函数，负责将数据写入文件。
        """
        try:
            with self._lock: # 确保线程安全
                with open(self.filepath, 'a', encoding='utf-8') as f:
                    for item in data_list:
                        f.write(json.dumps(item) + '\n')
        except IOError as e:
            print(f"Error: [AsyncRolloutSaver] 写入文件失败: {e}")
        except Exception as e:
            print(f"Error: [AsyncRolloutSaver] 发生未知错误: {e}")

    def save_long_rollouts(self, batch_data, long_rollout_indices, step: int):
        """
        公共接口，用于异步保存过长的 rollouts。
        :param batch_data: 包含所有 rollouts 的 DataProto 对象。
        :param long_rollout_indices: 过长 rollouts 在批次中的索引列表。
        :param step: 当前的训练步骤。
        """
        if not long_rollout_indices.numel(): # 如果没有需要保存的数据
            return

        # 准备数据
        data_to_save = self._prepare_data_for_serialization(batch_data, long_rollout_indices.tolist(), step)
        
        # 创建并启动一个守护线程来执行写入操作
        # 守护线程会在主程序退出时自动结束
        thread = threading.Thread(target=self._save_worker, args=(data_to_save,))
        thread.daemon = True
        thread.start()

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.config.algorithm.get("enable_model_merging", False):
            print("Initializing model merging baseline (post-checkpoint-load)...")
            self.actor_rollout_wg.initialize_merging_baseline()
            print("Model merging baseline initialized.")


        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        saver = AsyncRolloutSaver(filepath="/root/highspeedstorage/h800/data/long-rollouts/long_rollouts.jsonl")

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in (self.config.trainer.profile_steps or [])
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
    
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        max_len = self.config.data.max_response_length

                        # 只有在设置了有效阈值时才执行过滤
                        if max_len > 0:
                            # 假设长度可以通过 attention_mask 在维度1上的和来计算
                            # 请确保 'attention_mask' 是您数据结构中正确的键名
                            seq_lengths = new_batch.batch["attention_mask"].sum(dim=-1)

                            # 找到长度在阈值内的样本的索引
                            # 使用 PyTorch 的布尔索引，非常高效
                            long_rollout_indices = (seq_lengths >= max_len).nonzero(as_tuple=True)[0]

                            # 如果过滤后批次为空，则直接跳过，进行下一轮生成
                            if len(long_rollout_indices) > 0:
                                print(f"Found {long_rollout_indices.numel()} 个overlong rollouts (length >= {max_len}). will save it.")
                                saver.save_long_rollouts(new_batch, long_rollout_indices, self.global_steps)
            
                            if self.config.algorithm.filter_groups.filter_long_rollouts:
                                print("filter overlong rollouts")
                                
                                # 找到长度在阈值内的样本的索引 (用于过滤)
                                length_kept_indices = (seq_lengths < max_len).nonzero(as_tuple=True)[0]

                                # 如果过滤后批次为空，则直接跳过，进行下一轮生成
                                if len(length_kept_indices) == 0:
                                    print(f"长度过滤 (阈值={max_len}) 后，当前批次所有样本均被移除。继续生成...")
                                    progress_bar.update(1)
                                    continue # 跳到下一个生成批次

                                # 使用找到的索引来创建一个新的、更小的 new_batch
                                print(f'通过长度过滤: {len(seq_lengths)} -> {len(length_kept_indices)} 个响应')
                                new_batch = new_batch[length_kept_indices]
                            
                            else:
                                # 如果开关关闭，则不执行任何过滤操作
                                print("no filter overlong rollouts")
                                # new_batch 保持不变，包含所有序列

                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)
        
                    rewards_tensor = batch.batch["token_level_rewards"]
                    response_mask = batch.batch["response_mask"]

                    # Metric 1: Negative response ratio
                    # Calculate per-sequence reward, considering only the response part
                    sequence_rewards = (rewards_tensor * response_mask).sum(dim=-1)
                    total_sequences = sequence_rewards.size(0)
                    negative_sequences_mask = sequence_rewards < 0
                    num_negative_sequences = negative_sequences_mask.sum().item()
                    if total_sequences > 0:
                        negative_response_ratio = num_negative_sequences / total_sequences
                    else:
                        negative_response_ratio = 0.0
                    
                    # Metric 2: Negative token ratio
                    # Calculate total tokens in response
                    total_response_tokens = response_mask.sum().item()
                    if total_response_tokens > 0:
                        # Count negative reward tokens only within the response
                        num_tokens_in_negative_sequences = response_mask[negative_sequences_mask].sum().item()
                        negative_token_ratio = num_tokens_in_negative_sequences / total_response_tokens
                    else:
                        negative_token_ratio = 0.0

                    reward_stats_metrics = {
                        "actor/negative_response_ratio": negative_response_ratio,
                        "actor/negative_token_ratio": negative_token_ratio,
                    }
                    metrics.update(reward_stats_metrics)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    
                    if self.config.algorithm.get("enable_model_merging", False):
                        with marked_timer("model_merging", timing_raw, "purple"):
                            alpha = self.config.algorithm.get("model_merging_alpha", 0.0)
                            self.actor_rollout_wg.model_merging(alpha)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
