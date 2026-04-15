from __future__ import annotations
import math, json, os, time, re, random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from jinja2 import Template
import torch
from termcolor import cprint
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from llada.modeling_llada import LLaDAModelLM
import multiprocessing as mp

from omegaconf import DictConfig, ListConfig, OmegaConf


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    noise = (- torch.log(noise)) ** temperature
    return logits.exp() / noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@dataclass
class DiffusionOutput:
    sequences: torch.Tensor
    history:   List[torch.Tensor]
    nfe:       int
    mcmc_log:  List[List[dict]] | None = None



def compute_sequence_log_likelihood(
    model,
    sequences: torch.Tensor,
    start_idx: int,
    block_size: int,
    mask_id: int,
    horizon: int = 0,
) -> torch.Tensor:
    """
    Blockwise n-forward scoring (approximate), aligned with suffix-masked proposal:

      score 閳?鍗盻blocks 鍗盻{i in block} log p鑳?x_i | x_<start, mask_{start:end})

    where end = min(L, e + horizon) and we truncate beyond end.

    Notes:
      - For each block [s,e), we run ONE forward on a truncated sequence [0:end).
      - Input is: prefix fixed + suffix (start:end) fully masked.
      - We NEVER feed candidate suffix tokens back as input (avoid "seeing answers").
      - This is an approximation for bidirectional attention models due to truncation.
    """
    N, L = sequences.shape
    device = sequences.device
    if start_idx >= L:
        return torch.zeros(N, device=device, dtype=torch.float64)

    ll = torch.zeros(N, device=device, dtype=torch.float64)
    horizon = max(0, int(horizon))

    for s in range(start_idx, L, block_size):
        e = min(s + block_size, L)
        end = min(L, e + horizon)

        temp_x = sequences[:, :end].clone()
        temp_x[:, start_idx:end] = mask_id

        with torch.no_grad():
            logits = model(temp_x).logits

        lg = logits[:, s:e, :].float()
        tg = sequences[:, s:e]

        chosen = torch.gather(lg, dim=-1, index=tg.unsqueeze(-1)).squeeze(-1)
        logZ = torch.logsumexp(lg, dim=-1)
        ll += (chosen - logZ).sum(dim=1).to(torch.float64)

    return ll




def run_diffusion_generation_batch(
    model,
    x: torch.Tensor,
    start_idx: int,
    L0: int,
    steps: int,
    block_size: int,
    temperature: float,
    target: str,
    mask_id: int,
    further_horizon: int | None,
    use_cache: bool,
    unmask_threshold: float | None,
    intra_mcmc_steps: int = 0,
    intra_mcmc_alpha: float = 1.0,
    intra_num_candidates: int = 1,
) -> Tuple[torch.Tensor, int]:
    x = x.clone()
    N, total_len = x.shape
    x[:, start_idx:] = mask_id

    num_blocks = (total_len - L0) // block_size
    start_block_idx = (start_idx - L0) // block_size

    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (i < rem) for i in range(num_blocks)]

    nfe = 0
    cgws = further_horizon

    for blk in range(start_block_idx, num_blocks):
        s = L0 + blk * block_size
        e = min(L0 + (blk + 1) * block_size, total_len)
        block_len = e - s

        cur_steps = steps_per_block[blk]

        if cgws is not None:
            window_end = min(e + cgws, total_len)
            window_slice = slice(s, window_end)
        else:
            window_slice = None

        def run_block_once(
            current_x,
            step_offset: int = 0,
            base_log_probs: torch.Tensor | None = None,
            base_first_steps: torch.Tensor | None = None
        ):
            local_nfe = 0

            if base_log_probs is None:
                token_log_probs = torch.full((N, block_len), float("nan"), device=current_x.device, dtype=torch.float64)
            else:
                token_log_probs = base_log_probs.clone()

            if base_first_steps is None:
                first_steps = torch.full((N, block_len), -1, device=current_x.device, dtype=torch.long)
            else:
                first_steps = base_first_steps.clone()

            time_steps_left = max(1, cur_steps - step_offset)
            current_mask_count = (current_x[:, s:e] == mask_id).sum(dim=1).max().item()
            effective_steps = min(time_steps_left, max(int(current_mask_count), 1))

            num_transfer = get_num_transfer_tokens((current_x[:, s:e] == mask_id), effective_steps)

            pkv_local = None
            if use_cache:
                with torch.no_grad():
                    out_local = model(current_x, use_cache=True)
                pkv_full = out_local.past_key_values
                pkv_local = tuple(
                    tuple(t[:, :, :s] for t in layer) for layer in pkv_full
                )
                logits_initial = out_local.logits
            else:
                logits_initial = model(current_x, use_cache=False).logits

            mask_all = (current_x == mask_id)
            mask_all[:, e:] = 0

            x0, tr_idx, x0_p, x0_logp = get_transfer_index(
                logits_initial, temperature, target, mask_all,
                current_x, num_transfer[:, 0], unmask_threshold
            )

            if tr_idx.any():
                current_x[tr_idx] = x0[tr_idx].clone()
                block_tr = tr_idx[:, s:e]

                if block_tr.any():
                    update_mask = block_tr & (first_steps < 0)
                    if update_mask.any():
                        token_log_probs[update_mask] = x0_logp[:, s:e][update_mask]
                        first_steps[update_mask] = step_offset

            local_nfe += 1
            step_idx = step_offset + 1
            i_local = 1

            while True:
                local_nfe += 1
                step_transfer_idx = min(i_local, effective_steps - 1)

                mask_blk = (current_x[:, s:] == mask_id)
                mask_blk[:, block_len:] = 0

                if use_cache:
                    logits = model(current_x[:, s:], past_key_values=pkv_local, use_cache=True).logits
                    x0, tr_idx, x0_p, x0_logp = get_transfer_index(
                        logits, temperature, target,
                        mask_blk, current_x[:, s:], num_transfer[:, step_transfer_idx], unmask_threshold)

                    if tr_idx.any():
                        tmp = current_x[:, s:].clone()
                        tmp[tr_idx] = x0[tr_idx].clone()
                        current_x[:, s:] = tmp

                        block_tr = tr_idx[:, :block_len]
                        if block_tr.any():
                            update_mask = block_tr & (first_steps < 0)
                            if update_mask.any():
                                token_log_probs[update_mask] = x0_logp[:, :block_len][update_mask]
                                first_steps[update_mask] = step_idx
                else:
                    logits = model(current_x, use_cache=False).logits
                    logits = logits[:, s:]
                    x0, tr_idx, x0_p, x0_logp = get_transfer_index(
                        logits, temperature, target,
                        mask_blk, current_x[:, s:], num_transfer[:, step_transfer_idx], unmask_threshold)

                    if tr_idx.any():
                        tmp = current_x[:, s:].clone()
                        tmp[tr_idx] = x0[tr_idx].clone()
                        current_x[:, s:] = tmp

                        block_tr = tr_idx[:, :block_len]
                        if block_tr.any():
                            update_mask = block_tr & (first_steps < 0)
                            if update_mask.any():
                                token_log_probs[update_mask] = x0_logp[:, :block_len][update_mask]
                                first_steps[update_mask] = step_idx

                if (current_x[:, s:e] == mask_id).sum() == 0:
                    break

                i_local += 1
                step_idx += 1

            return current_x, token_log_probs, first_steps, local_nfe

        x[:, s:e] = mask_id
        x, token_log_probs, first_steps, nfe_inc = run_block_once(x)
        nfe += nfe_inc

        current_score = torch.nan_to_num(token_log_probs, nan=-1e9).sum(dim=1)

        current_tokens = x[:, s:e].clone()
        current_log_probs = token_log_probs.clone()
        current_first_steps = first_steps.clone()

        if intra_mcmc_steps > 0:
            for mcmc_iter in range(intra_mcmc_steps):
                if intra_num_candidates <= 1:
                    backup_tokens = current_tokens.clone()
                    backup_log_probs = current_log_probs.clone()
                    backup_first_steps = current_first_steps.clone()
                    backup_score = current_score.clone()

                    if cur_steps > 1:
                        rollback_t = random.randint(0, cur_steps - 1)
                        rollback_mask = (current_first_steps > rollback_t) & (current_first_steps >= 0)
                        if rollback_mask.any():
                            current_tokens[rollback_mask] = mask_id
                            current_log_probs[rollback_mask] = float("nan")
                            current_first_steps[rollback_mask] = -1
                            x[:, s:e] = current_tokens
                    else:
                        rollback_t = 0

                    x, new_log_probs, new_first_steps, nfe_inc = run_block_once(
                        x,
                        step_offset=rollback_t + 1,
                        base_log_probs=current_log_probs,
                        base_first_steps=current_first_steps
                    )
                    nfe += nfe_inc
                    new_score = torch.nan_to_num(new_log_probs, nan=-1e9).sum(dim=1)

                    ratio = (intra_mcmc_alpha - 1.0) * (new_score - current_score)
                    rand = torch.log(torch.rand_like(ratio).clamp_min(1e-12))
                    accept_mask = (ratio >= 0) | (rand < ratio)

                    for b in range(N):
                        if accept_mask[b]:
                            current_tokens[b] = x[b, s:e]
                            current_log_probs[b] = new_log_probs[b]
                            current_first_steps[b] = new_first_steps[b]
                            current_score[b] = new_score[b]
                        else:
                            current_tokens[b] = backup_tokens[b]
                            current_log_probs[b] = backup_log_probs[b]
                            current_first_steps[b] = backup_first_steps[b]
                            current_score[b] = backup_score[b]
                            x[b, s:e] = current_tokens[b]
                else:
                    K = intra_num_candidates
                    backup_tokens = current_tokens.clone()
                    backup_log_probs = current_log_probs.clone()
                    backup_first_steps = current_first_steps.clone()
                    backup_score = current_score.clone()

                    if cur_steps > 1:
                        rollback_t = random.randint(0, cur_steps - 1)
                    else:
                        rollback_t = 0

                    candidate_tokens = current_tokens.repeat_interleave(K, dim=0)
                    candidate_log_probs = current_log_probs.repeat_interleave(K, dim=0)
                    candidate_first_steps = current_first_steps.repeat_interleave(K, dim=0)
                    x_candidates = x.repeat_interleave(K, dim=0)

                    if cur_steps > 1:
                        rollback_mask = (current_first_steps > rollback_t) & (current_first_steps >= 0)
                        rollback_mask_candidates = rollback_mask.repeat_interleave(K, dim=0)
                        candidate_tokens[rollback_mask_candidates] = mask_id
                        candidate_log_probs[rollback_mask_candidates] = float("nan")
                        candidate_first_steps[rollback_mask_candidates] = -1

                    x_candidates[:, s:e] = candidate_tokens
                    x_candidates, new_log_probs_cands, new_first_steps_cands, nfe_inc = run_block_once(
                        x_candidates,
                        step_offset=rollback_t + 1,
                        base_log_probs=candidate_log_probs,
                        base_first_steps=candidate_first_steps
                    )
                    nfe += nfe_inc

                    new_scores_cands = torch.nan_to_num(new_log_probs_cands, nan=-1e9).sum(dim=1)
                    new_scores_cands = new_scores_cands.view(N, K)
                    new_log_probs_cands = new_log_probs_cands.view(N, K, block_len)
                    new_first_steps_cands = new_first_steps_cands.view(N, K, block_len)
                    cand_tokens_reshaped = x_candidates[:, s:e].view(N, K, block_len)

                    log_weights = (intra_mcmc_alpha - 1.0) * (new_scores_cands - backup_score.unsqueeze(1))
                    max_log_w = torch.max(log_weights, dim=1, keepdim=True).values
                    exp_log_weights = torch.exp(log_weights - max_log_w)

                    sum_exp_log_weights = exp_log_weights.sum(dim=1, keepdim=True)
                    probs = exp_log_weights / sum_exp_log_weights

                    idx_tensor = torch.multinomial(probs.float(), num_samples=1).squeeze(-1)

                    sum_weights = exp_log_weights.sum(dim=1)
                    chosen_weights = exp_log_weights.gather(1, idx_tensor.unsqueeze(1)).squeeze(1)
                    W_fwd = sum_weights
                    W_bwd = 1.0 + (sum_weights - chosen_weights)
                    acc_prob = torch.minimum(W_fwd / W_bwd, torch.ones_like(W_fwd))
                    rand_acc = torch.rand_like(acc_prob)
                    accept_mask = acc_prob > rand_acc

                    for b in range(N):
                        if accept_mask[b]:
                            chosen_idx = idx_tensor[b].item()
                            current_tokens[b] = cand_tokens_reshaped[b, chosen_idx]
                            current_log_probs[b] = new_log_probs_cands[b, chosen_idx]
                            current_first_steps[b] = new_first_steps_cands[b, chosen_idx]
                            current_score[b] = new_scores_cands[b, chosen_idx]
                            x[b, s:e] = current_tokens[b]
                        else:
                            current_tokens[b] = backup_tokens[b]
                            current_log_probs[b] = backup_log_probs[b]
                            current_first_steps[b] = backup_first_steps[b]
                            current_score[b] = backup_score[b]
                            x[b, s:e] = current_tokens[b]

        x[:, s:e] = current_tokens

    return x, nfe


@torch.no_grad()
def generate_with_prefix_cache(
        model, prompt,
        steps, gen_length, block_length, temperature,
        target, mask_id, further_horizon, use_cache, unmask_threshold,
        mcmc_alpha: float = 1.0, mcmc_steps: int = 0,
        mcmc_log: bool = False,
        num_candidates: int = 1,
        intra_mcmc_steps: int = 0,
        intra_mcmc_alpha: float = 1.0,
        intra_num_candidates: int = 1,
    ) -> DiffusionOutput:

    B, L0 = prompt.shape
    L_total = L0 + gen_length

    x = torch.full((B, L_total), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :L0] = prompt

    hist = []
    mcmc_records = [[] for _ in range(B)] if mcmc_log else None
    total_nfe = 0

    cprint(f"Phase 1: Initial Generation (Intra-MTM steps={intra_mcmc_steps}, K={intra_num_candidates})...", "cyan")
    x, nfe = run_diffusion_generation_batch(
        model, x, start_idx=L0, L0=L0, steps=steps, block_size=block_length,
        temperature=temperature, target=target, mask_id=mask_id,
        further_horizon=further_horizon, use_cache=use_cache, unmask_threshold=unmask_threshold,
        intra_mcmc_steps=intra_mcmc_steps,
        intra_mcmc_alpha=intra_mcmc_alpha,
        intra_num_candidates=intra_num_candidates,
    )
    total_nfe += nfe
    hist.append(x.clone().cpu())

    if mcmc_steps > 0:
        K = max(1, int(num_candidates))
        cprint(f"Phase 2: Inter-Block MTM Refinement (Steps={mcmc_steps}, K={K})...", "cyan")

        num_gen_blocks = gen_length // block_length

        for step in range(mcmc_steps):
            m = random.randint(0, num_gen_blocks - 1)
            mask_start_idx = L0 + m * block_length

            if K == 1:
                y, nfe_fwd = run_diffusion_generation_batch(
                    model, x.clone(), start_idx=mask_start_idx, L0=L0, steps=steps,
                    block_size=block_length, temperature=temperature, target=target, mask_id=mask_id,
                    further_horizon=further_horizon, use_cache=use_cache, unmask_threshold=unmask_threshold,
                    intra_mcmc_steps=intra_mcmc_steps,
                    intra_mcmc_alpha=intra_mcmc_alpha,
                    intra_num_candidates=intra_num_candidates,
                )
                total_nfe += nfe_fwd

                ll_y = compute_sequence_log_likelihood(model, y, mask_start_idx, block_length, mask_id)
                ll_x = compute_sequence_log_likelihood(model, x, mask_start_idx, block_length, mask_id)

                log_W_forward = (mcmc_alpha - 1.0) * ll_y
                log_W_backward = (mcmc_alpha - 1.0) * ll_x

                log_rho = log_W_forward - log_W_backward
                accept_prob = torch.exp(log_rho).clamp(max=1.0)
                accept_mask = torch.rand(B, device=x.device) < accept_prob

                x[accept_mask] = y[accept_mask]
                hist.append(x.clone().cpu())
                continue

            x_expanded = expand_inputs_for_candidates(x, K)

            y_candidates, nfe_fwd = run_diffusion_generation_batch(
                model, x_expanded, start_idx=mask_start_idx, L0=L0, steps=steps,
                block_size=block_length, temperature=temperature, target=target, mask_id=mask_id,
                further_horizon=further_horizon, use_cache=use_cache, unmask_threshold=unmask_threshold,
                intra_mcmc_steps=intra_mcmc_steps,
                intra_mcmc_alpha=intra_mcmc_alpha,
                intra_num_candidates=intra_num_candidates,
            )
            total_nfe += nfe_fwd

            ll_y = compute_sequence_log_likelihood(model, y_candidates, mask_start_idx, block_length, mask_id)
            log_w_y = (mcmc_alpha - 1.0) * ll_y

            log_w_y_reshaped = log_w_y.view(B, K)
            probs_y = F.softmax(log_w_y_reshaped, dim=-1)
            selected_indices = torch.multinomial(probs_y, 1).squeeze(-1)

            batch_indices = torch.arange(B, device=x.device) * K + selected_indices
            y_star = y_candidates[batch_indices]

            log_W_forward = torch.logsumexp(log_w_y_reshaped, dim=-1)

            z_inputs = expand_inputs_for_candidates(y_star, K)
            for b in range(B):
                z_inputs[b * K] = x[b]

            indices_to_generate = []
            for b in range(B):
                for k in range(1, K):
                    indices_to_generate.append(b * K + k)

            if len(indices_to_generate) > 0:
                z_subset = z_inputs[indices_to_generate]
                z_subset_gen, nfe_bwd = run_diffusion_generation_batch(
                    model, z_subset, start_idx=mask_start_idx, L0=L0, steps=steps,
                    block_size=block_length, temperature=temperature, target=target, mask_id=mask_id,
                    further_horizon=further_horizon, use_cache=use_cache, unmask_threshold=unmask_threshold,
                    intra_mcmc_steps=intra_mcmc_steps,
                    intra_mcmc_alpha=intra_mcmc_alpha,
                    intra_num_candidates=intra_num_candidates,
                )
                total_nfe += nfe_bwd
                z_inputs[indices_to_generate] = z_subset_gen

            ll_z = compute_sequence_log_likelihood(model, z_inputs, mask_start_idx, block_length, mask_id)
            log_w_z = (mcmc_alpha - 1.0) * ll_z
            log_W_backward = torch.logsumexp(log_w_z.view(B, K), dim=-1)

            log_rho = log_W_forward - log_W_backward
            accept_prob = torch.exp(log_rho).clamp(max=1.0)
            accept_mask = torch.rand(B, device=x.device) < accept_prob

            x[accept_mask] = y_star[accept_mask]
            hist.append(x.clone().cpu())

    return DiffusionOutput(sequences=x, history=hist, nfe=total_nfe, mcmc_log=mcmc_records)


def expand_inputs_for_candidates(inputs: torch.Tensor, K: int) -> torch.Tensor:
    if K == 1:
        return inputs
    return inputs.repeat_interleave(K, dim=0).contiguous()


def get_transfer_index(logits, temperature, target, mask_index, x, num_transfer_tokens, threshold=None):
    """
    Returns:
      x0: tokens after applying masked positions
      transfer_index: which positions to transfer (fill) this step
      x0_p: selection score (confidence/margin/entropy/random)
      x0_logp: 閴?true logprob of chosen token under softmax(logits)
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if target == 'confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif target == 'margin_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        top2 = torch.topk(p, 2, dim=-1).values
        x0_p = top2[..., 0] - top2[..., 1]
    elif target == 'neg_entropy':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
    elif target == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(target)

    logp_vocab = F.log_softmax(logits.to(torch.float64), dim=-1)
    x0_logp = torch.squeeze(
        torch.gather(logp_vocab, dim=-1, index=torch.unsqueeze(x0, -1)), -1
    )

    x0 = torch.where(mask_index, x0, x)

    if threshold is not None:
        selected = mask_index & (x0_p >= threshold)
        has_mask = mask_index.any(dim=-1)
        none_sel = (~selected.any(dim=-1)) & has_mask
        if none_sel.any():
            masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
            best_idx = masked_scores.argmax(dim=-1)
            rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
            selected[rows, best_idx[rows]] = True
        return x0, selected, x0_p, x0_logp

    confidence = x0_p.masked_fill(~mask_index, float("-inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j].item() if torch.is_tensor(num_transfer_tokens[j]) else num_transfer_tokens[j])
        if k <= 0:
            continue
        masked_pos = torch.nonzero(mask_index[j], as_tuple=False).squeeze(-1)
        if masked_pos.numel() == 0:
            continue
        if k >= masked_pos.numel():
            sel = masked_pos
        else:
            scores = confidence[j, masked_pos]
            _, top_idx = torch.topk(scores, k=k)
            sel = masked_pos[top_idx]
        transfer_index[j, sel] = True
    return x0, transfer_index, x0_p, x0_logp


def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
    L = history[0].shape[1]
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)

    for t, snap in enumerate(history, start=0):
        cur = snap[sample_idx]
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
        if (step_map == 0).sum() == 0:
            break
    return step_map


import sys
from tqdm import tqdm


def worker(pretrained_model, rank, prompts, orig_idx, seq_dict, step_dict, detail_dict, mcmc_dict,
           target_indices, batch_size, config, log_mcmc):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model_gpu = (LLaDAModelLM
                 .from_pretrained(pretrained_model,
                                  trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)
                 .to(device)
                 .eval())

    tokenizer_gpu = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    use_tqdm = sys.stdout.isatty()
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"GPU {rank}",
        position=rank if use_tqdm else 0,
        leave=use_tqdm,
        dynamic_ncols=True,
        disable=not use_tqdm,
    ):
        batch_prompts = prompts[start:start+batch_size]
        batch_idxs    = orig_idx[start:start+batch_size]

        enc = tokenizer_gpu(batch_prompts,
                            padding=True,
                            return_tensors="pt", padding_side="left")
        input_ids = enc["input_ids"].to(device)
        mask_id = tokenizer_gpu.encode('<|mdm_mask|>')[0]

        if config.rollout.use_cache == False:
            config.rollout.further_horizon = None

        if config.rollout.remasking_strategy == "low_confidence_static":
            unmask_threshold = None
        else:
            unmask_threshold = config.rollout.dynamic_threshold

        mcmc_alpha = getattr(config.rollout, "mcmc_alpha", 1.0)
        mcmc_steps = getattr(config.rollout, "mcmc_steps", 0)
        num_candidates = getattr(config.rollout, "num_candidates", 1)

        intra_mcmc_steps = getattr(config.rollout, "intra_mcmc_steps", 0)
        intra_mcmc_alpha = getattr(config.rollout, "intra_mcmc_alpha", mcmc_alpha)
        intra_num_candidates = getattr(config.rollout, "intra_num_candidates", num_candidates)

        out = generate_with_prefix_cache(
            model_gpu, input_ids,
            steps=config.rollout.steps, gen_length=config.rollout.max_gen_length,
            block_length=config.rollout.block_size, temperature=config.rollout.temperature,
            target=config.rollout.target, mask_id=mask_id, further_horizon=config.rollout.further_horizon,
            use_cache=config.rollout.use_cache, unmask_threshold=unmask_threshold,
            mcmc_alpha=mcmc_alpha, mcmc_steps=mcmc_steps,
            mcmc_log=log_mcmc, num_candidates=num_candidates,
            intra_mcmc_steps=intra_mcmc_steps,
            intra_mcmc_alpha=intra_mcmc_alpha,
            intra_num_candidates=intra_num_candidates,
        )
        out.sequences = out.sequences.cpu()
        torch.cuda.empty_cache()

        seq_ids = out.sequences[:, input_ids.shape[1]:].tolist()
        texts  = tokenizer_gpu.batch_decode(
            seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        for i, idx in enumerate(batch_idxs):
            m = denoise_step_map(out.history, mask_id=mask_id, sample_idx=i)
            step_map = m[input_ids.shape[1]:].tolist()
            seq_dict[idx]  = texts[i]
            step_dict[idx] = step_map
            if idx in target_indices:
                history_outputs = []
                for t, snap in enumerate(out.history):
                    step_tokens = snap[i, input_ids.shape[1]:].tolist()
                    step_text = tokenizer_gpu.decode(
                        step_tokens,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
                    history_outputs.append({"step": t, "text": step_text})
                detail_dict[idx] = {
                    "prompt": batch_prompts[i],
                    "final_text": texts[i],
                    "step_map": step_map,
                    "history": history_outputs,
                }
            if log_mcmc and out.mcmc_log is not None:
                mcmc_dict[idx] = {
                    "prompt": batch_prompts[i],
                    "comparisons": out.mcmc_log[i],
                }
        torch.cuda.empty_cache()


def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output


def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"
    i = start + len(tag)
    depth = 1
    buf = []
    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def get_prompt(data_i):
    return Template(system_prompts).render(problem=data_i["question"])


def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]


if __name__ == "__main__":
    config = get_config()
    mp.set_start_method("spawn", force=True)

    k_sample = config.rollout.num_response_per_task
    batch_size = config.rollout.batch_size
    project_name = config.experiment.project

    system_prompts = """<|startoftext|><|start_header_id|>user<|end_header_id|>You need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""

    code_eval = False
    dataset = config.dataset.eval_dataset
    pretrained_model = config.model
    if config.dataset.data_type == "code":
        code_eval = True
        system_prompts_function = '''<|startoftext|><|start_header_id|>user<|end_header_id|>{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n'''
        system_prompts_stdio = '''<|startoftext|><|start_header_id|>user<|end_header_id|>This is the problem:\n{{problem}}\n You should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n'''
    elif config.dataset.data_type == "option":
        system_prompts = '''<|startoftext|><|start_header_id|>user<|end_header_id|>This is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only閳ユ攺o other character) in \\boxed{}. <|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n'''

    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", dataset + ".json")

    with open(data_path, "r") as f:
        data = json.load(f)

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if num_node > 1:
        data = get_data_chunk(data, num_node, node_index)
    eval_first_n = getattr(config.dataset, "eval_first_n", None)
    if eval_first_n is None:
        eval_first_n = getattr(config.experiment, "eval_first_n", None)
    if eval_first_n is not None and int(eval_first_n) > 0:
        data = data[: int(eval_first_n)]
    log_mcmc = eval_first_n is not None and int(eval_first_n) == 10

    num = len(data)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        if code_eval:
            if data[i]["test_method"] == "stdio":
                system_prompts = system_prompts_stdio
                prefix_list = prefix_list + [None] * k_sample
            else:
                system_prompts = system_prompts_function + data[i]["prefix"]
                prefix_list = prefix_list + [data[i]["prefix"]] * k_sample
        generation_prompts = generation_prompts + [get_prompt(data[i])] * k_sample
        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = get_prompt(data[i])

    cprint("start generation...", "green")
    gen_start_time = time.time()

    all_prompts = generation_prompts
    N = len(all_prompts)
    shuffled_idx = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]

    n_gpu = torch.cuda.device_count()
    assert n_gpu > 1, "need >=2 GPUs for parallel inference"

    def split_even(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    prompt_chunks = split_even(shuffled_prompts, n_gpu)
    idx_chunks = split_even(shuffled_idx, n_gpu)

    manager = mp.Manager()
    seq_dict = manager.dict()
    step_dict = manager.dict()
    detail_dict = manager.dict()
    mcmc_dict = manager.dict()
    procs = []
    target_indices = set(range(min(5, N)))

    for rk in range(n_gpu):
        p = mp.Process(
            target=worker,
            args=(
                pretrained_model, rk,
                prompt_chunks[rk],
                idx_chunks[rk],
                seq_dict,
                step_dict,
                detail_dict,
                mcmc_dict,
                target_indices,
                batch_size,
                config,
                log_mcmc
            )
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    restored_outputs = [seq_dict[i] for i in range(N)]
    restored_step_maps = [step_dict[i] for i in range(N)]

    cprint("generation job done!", "green")
    gen_elapsed = time.time() - gen_start_time

    def get_token_lengths(strings, tokenizer):
        pad_token = tokenizer.pad_token
        escaped = re.escape(pad_token)
        pattern = rf"(?:{escaped})+"
        remove_pattern = escaped
        collapse_re = re.compile(pattern)
        lengths = []
        for s in strings:
            s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
            s_clean = re.sub(remove_pattern, '', s_clean)
            lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
        return lengths

    response_length = get_token_lengths(restored_outputs, tokenizer)
    mean_response_length = sum(response_length) / len(response_length)
    cprint(f"avg_response_length: {mean_response_length:.2f}", "green")
    cprint(f"generation_time_sec: {gen_elapsed:.2f}", "green")

    detail_entries = []
    for idx in sorted(detail_dict.keys()):
        entry = detail_dict[idx]
        entry["shuffled_index"] = idx
        entry["original_index"] = shuffled_idx[idx]
        detail_entries.append(entry)
    if detail_entries:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        detail_file_name = "../" + project_name + "/temp_data/detail-" + outputs_name + "-" + timestamp + ".json"
        os.makedirs(os.path.dirname(detail_file_name), exist_ok=True)
        with open(detail_file_name, "w", encoding="utf-8") as f:
            json.dump(detail_entries, f, indent=2, ensure_ascii=False)
    if log_mcmc and len(mcmc_dict) > 0:
        mcmc_entries = []
        for idx in sorted(mcmc_dict.keys()):
            entry = mcmc_dict[idx]
            entry["shuffled_index"] = idx
            entry["original_index"] = shuffled_idx[idx]
            mcmc_entries.append(entry)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        mcmc_file_name = "../" + project_name + "/temp_data/mcmc-compare-" + outputs_name + "-" + timestamp + ".json"
        os.makedirs(os.path.dirname(mcmc_file_name), exist_ok=True)
        with open(mcmc_file_name, "w", encoding="utf-8") as f:
            json.dump(mcmc_entries, f, indent=2, ensure_ascii=False)

    i = 0
    for full_output in restored_outputs:
        if code_eval:
            if data[int(i/k_sample)]["test_method"] == "function":
                extracted_output = extract_code(prefix_list[i] + full_output)
            else:
                extracted_output = extract_code(full_output)
        else:
            extracted_output = extract_final_boxed_answer(full_output)
        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        data[index_i]["step_map"].append(restored_step_maps[i])
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])
        i += 1

    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
