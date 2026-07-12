from __future__ import annotations

import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from dataevol.experiments.rl_socket_trainer import (
    PPOConfig,
    StatefulSampler,
    ToyCausalPolicy,
    ToyUniformLoRAPolicy,
    categorical_kl,
    clone_frozen,
    completion_logprobs,
    load_checkpoint,
    make_rollout_batch,
    masked_mean,
    parameter_hash,
    ppo_loss,
    sample_completion,
    save_checkpoint,
    tree_allclose,
)


def test_policy_logprob_is_prompt_conditioned_and_completion_only() -> None:
    mx.random.seed(7)
    model = ToyCausalPolicy()
    tokens = mx.array([[1, 2, 6], [1, 4, 6]], dtype=mx.int32)
    logprobs, mask = completion_logprobs(model(tokens), tokens, mx.array([2, 2]), mx.array([1, 1]))
    mx.eval(logprobs)

    assert bool(mask[0, 0]) is False
    assert bool(mask[0, 1]) is True
    assert float(logprobs[0, 1]) != float(logprobs[1, 1])


def test_prompt_and_padding_positions_are_masked_but_prompt_supplies_context() -> None:
    mx.random.seed(8)
    model = ToyCausalPolicy()
    tokens = mx.array([[1, 2, 3, 6, 0], [1, 4, 5, 6, 0]], dtype=mx.int32)
    prompt_lengths = mx.array([3, 3])
    completion_lengths = mx.array([1, 1])
    logits = model(tokens)

    def loss(candidate_logits: mx.array) -> mx.array:
        logprobs, mask = completion_logprobs(candidate_logits, tokens, prompt_lengths, completion_lengths)
        return -masked_mean(logprobs, mask)

    gradients = mx.grad(loss)(logits)
    mx.eval(gradients)
    assert np.allclose(np.asarray(gradients[:, :2]), 0.0)
    assert not np.allclose(np.asarray(gradients[:, 2]), 0.0)
    assert np.allclose(np.asarray(gradients[:, 3:]), 0.0)

    logprobs, _ = completion_logprobs(logits, tokens, prompt_lengths, completion_lengths)
    assert float(logprobs[0, 2]) != float(logprobs[1, 2])


def test_old_policy_and_reference_policy_are_separate() -> None:
    mx.random.seed(9)
    policy = ToyCausalPolicy()
    reference = clone_frozen(policy)
    tokens = mx.array([[1, 2, 6], [1, 2, 7]], dtype=mx.int32)
    batch = make_rollout_batch(
        policy, reference, tokens, mx.array([2, 2]), mx.array([1, 1]), mx.array([1.0, 0.0])
    )

    assert policy is not reference
    assert batch.old_logprobs is not batch.reference_logprobs
    before = np.asarray(batch.old_logprobs).copy()
    perturbation = mx.zeros_like(policy.output.weight)
    perturbation[6, :] = 0.5
    policy.output.weight = policy.output.weight + perturbation
    current, _ = completion_logprobs(policy(tokens), tokens, batch.prompt_lengths, batch.completion_lengths)
    ratio = mx.exp(current - batch.old_logprobs)
    mx.eval(ratio)
    assert np.array_equal(np.asarray(batch.old_logprobs), before)
    assert not np.allclose(np.asarray(ratio), 1.0)


def test_kl_responds_to_policy_change_and_beta() -> None:
    mx.random.seed(10)
    policy = ToyCausalPolicy()
    reference = clone_frozen(policy)
    tokens = mx.array([[1, 2, 6], [1, 2, 7]], dtype=mx.int32)
    batch = make_rollout_batch(
        policy, reference, tokens, mx.array([2, 2]), mx.array([1, 1]), mx.array([1.0, 0.0])
    )
    initial_loss, initial_metrics = ppo_loss(policy(tokens), batch, PPOConfig(beta=0.5))
    assert abs(float(initial_metrics["kl"])) < 1e-6

    perturbation = mx.zeros_like(policy.output.weight)
    perturbation[6, :] = 0.5
    policy.output.weight = policy.output.weight + perturbation
    no_kl, no_kl_metrics = ppo_loss(policy(tokens), batch, PPOConfig(beta=0.0), include_kl=False)
    beta_low, low_metrics = ppo_loss(policy(tokens), batch, PPOConfig(beta=0.1))
    beta_high, high_metrics = ppo_loss(policy(tokens), batch, PPOConfig(beta=0.5))
    mx.eval(initial_loss, no_kl, beta_low, beta_high)

    assert float(low_metrics["kl"]) > 0
    assert np.isclose(float(low_metrics["policy_loss"]), float(no_kl_metrics["policy_loss"]))
    assert float(beta_high) > float(beta_low) > float(no_kl)
    assert np.isclose(float(low_metrics["kl"]), float(high_metrics["kl"]))


def test_rewarded_completion_increases_relative_probability() -> None:
    mx.random.seed(11)
    policy = ToyCausalPolicy()
    reference = clone_frozen(policy)
    tokens = mx.array([[1, 2, 6], [1, 2, 7]], dtype=mx.int32)
    prompt_lengths = mx.array([2, 2])
    completion_lengths = mx.array([1, 1])
    batch = make_rollout_batch(
        policy, reference, tokens, prompt_lengths, completion_lengths, mx.array([1.0, 0.0])
    )
    before, _ = completion_logprobs(policy(tokens), tokens, prompt_lengths, completion_lengths)
    before_margin = float(before[0, 1] - before[1, 1])
    optimizer = optim.AdamW(learning_rate=0.05, weight_decay=0.0)

    def objective() -> mx.array:
        return ppo_loss(policy(tokens), batch, PPOConfig(beta=0.0), include_kl=False)[0]

    loss_and_grad = nn.value_and_grad(policy, objective)
    loss, gradients = loss_and_grad()
    optimizer.update(policy, gradients)
    mx.eval(policy.parameters(), optimizer.state, loss)
    after, _ = completion_logprobs(policy(tokens), tokens, prompt_lengths, completion_lengths)
    after_margin = float(after[0, 1] - after[1, 1])
    assert after_margin > before_margin


def test_sampling_diversity_reset_and_checkpoint_reproduce_next_sample(tmp_path) -> None:
    mx.random.seed(12)
    model = ToyCausalPolicy()
    optimizer = optim.AdamW(0.01)
    sampler = StatefulSampler(1200)
    initial_state = sampler.state_dict()
    samples = [sample_completion(model, [1, 2], sampler, max_tokens=2, temperature=1.2) for _ in range(8)]
    assert len({tuple(sample) for sample in samples}) > 1
    assert sampler.counter == 16

    reset = StatefulSampler(0)
    reset.load_state_dict(initial_state)
    assert samples == [sample_completion(model, [1, 2], reset, max_tokens=2, temperature=1.2) for _ in range(8)]

    checkpoint = save_checkpoint(
        tmp_path,
        update=0,
        model=model,
        optimizer=optimizer,
        gradient_accumulator={},
        sampler=sampler,
        scheduler_state={"update": 0},
        training_state={"update": 0},
        config={"optimizer": {"name": "AdamW", "learning_rate": 0.01}},
    )
    uninterrupted_next = sample_completion(model, [1, 2], sampler, max_tokens=3, temperature=0.9)
    restored_model = ToyCausalPolicy()
    restored_optimizer = optim.AdamW(0.01)
    restored_sampler = StatefulSampler(1)
    load_checkpoint(checkpoint, model=restored_model, optimizer=restored_optimizer, sampler=restored_sampler)
    restored_next = sample_completion(restored_model, [1, 2], restored_sampler, max_tokens=3, temperature=0.9)
    assert restored_next == uninterrupted_next


def test_positive_control_uniform_lora_improves_easy_behavior() -> None:
    mx.random.seed(13)
    model = ToyUniformLoRAPolicy()
    tokens = mx.array([[1, 2, 6], [1, 3, 7], [1, 4, 8], [1, 5, 9]], dtype=mx.int32)
    prompt_lengths = mx.array([2, 2, 2, 2])
    completion_lengths = mx.array([1, 1, 1, 1])

    def accuracy() -> float:
        predictions = mx.argmax(model(tokens)[:, 1, :], axis=-1)
        return float(mx.mean((predictions == tokens[:, 2]).astype(mx.float32)))

    before = accuracy()
    optimizer = optim.AdamW(learning_rate=0.05, weight_decay=0.0)

    def objective() -> mx.array:
        logprobs, mask = completion_logprobs(model(tokens), tokens, prompt_lengths, completion_lengths)
        return -masked_mean(logprobs, mask)

    loss_and_grad = nn.value_and_grad(model, objective)
    for _ in range(80):
        loss, gradients = loss_and_grad()
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
    after = accuracy()
    assert after - before >= 0.50
    assert after >= 0.75


def test_resume_50_plus_50_matches_100_updates(tmp_path) -> None:
    uninterrupted = _run_toy_rl(100, tmp_path / "uninterrupted")
    first_half = _run_toy_rl(50, tmp_path / "resumed", checkpoint=True)
    resumed = _run_toy_rl(100, tmp_path / "resumed", resume_from=first_half["checkpoint"])

    assert uninterrupted["parameter_hash"] == resumed["parameter_hash"]
    assert tree_allclose(uninterrupted["optimizer_state"], resumed["optimizer_state"], tolerance=0.0)
    assert uninterrupted["next_prompt"] == resumed["next_prompt"]
    assert uninterrupted["next_sample"] == resumed["next_sample"]
    assert uninterrupted["reward_history"] == resumed["reward_history"]
    assert uninterrupted["behavioral_score"] == resumed["behavioral_score"]


def _run_toy_rl(
    stop_update: int,
    root: Path,
    *,
    checkpoint: bool = False,
    resume_from: str | Path | None = None,
) -> dict[str, object]:
    mx.random.seed(44)
    random.seed(44)
    np.random.seed(44)
    policy = ToyCausalPolicy()
    reference = clone_frozen(policy)
    optimizer = optim.AdamW(learning_rate=0.015, weight_decay=0.0)
    sampler = StatefulSampler(4400)
    update = 0
    reward_history: list[list[float]] = []
    if resume_from is not None:
        loaded = load_checkpoint(resume_from, model=policy, optimizer=optimizer, sampler=sampler)
        update = int(loaded["training_state"]["update"])
        reward_history = list(loaded["training_state"]["reward_history"])
    prompts = ([1, 2], [1, 3], [1, 4])
    targets = {2: 6, 3: 7, 4: 8}
    config = PPOConfig(beta=0.01, learning_rate=0.015)

    while update < stop_update:
        prompt = list(random.choice(prompts))
        target = targets[prompt[-1]]
        wrong = 9 if target != 9 else 10
        sampled = sample_completion(policy, prompt, sampler, max_tokens=1, temperature=0.9)[0]
        rewards = [1.0 if sampled == target else 0.0, 0.0]
        reward_history.append(rewards)
        tokens = mx.array([prompt + [target], prompt + [wrong]], dtype=mx.int32)
        batch = make_rollout_batch(
            policy, reference, tokens, mx.array([2, 2]), mx.array([1, 1]), mx.array([1.0, 0.0])
        )

        def objective() -> mx.array:
            return ppo_loss(policy(tokens), batch, config)[0]

        loss_and_grad = nn.value_and_grad(policy, objective)
        loss, gradients = loss_and_grad()
        optimizer.update(policy, gradients)
        mx.eval(policy.parameters(), optimizer.state, loss)
        update += 1

    checkpoint_path = None
    if checkpoint:
        checkpoint_path = save_checkpoint(
            root,
            update=update,
            model=policy,
            optimizer=optimizer,
            gradient_accumulator={},
            sampler=sampler,
            scheduler_state={"update": update},
            training_state={"update": update, "reward_history": reward_history},
            config={"optimizer": {"name": "AdamW", "learning_rate": 0.015, "weight_decay": 0.0}},
        )

    next_prompt = list(random.choice(prompts))
    next_sample = sample_completion(policy, next_prompt, sampler, max_tokens=2, temperature=0.9)
    evaluation_tokens = mx.array([list(prompt) for prompt in prompts], dtype=mx.int32)
    predictions = mx.argmax(policy(evaluation_tokens)[:, -1, :], axis=-1)
    expected = mx.array([targets[prompt[-1]] for prompt in prompts])
    score = float(mx.mean((predictions == expected).astype(mx.float32)))
    return {
        "checkpoint": checkpoint_path,
        "parameter_hash": parameter_hash(policy),
        "optimizer_state": optimizer.state,
        "next_prompt": next_prompt,
        "next_sample": next_sample,
        "reward_history": reward_history,
        "behavioral_score": score,
    }
