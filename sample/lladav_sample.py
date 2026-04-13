from __future__ import annotations
import json, os
import sys
from dataclasses import dataclass
from typing import List
import random
from jinja2 import Template
import torch
from termcolor import cprint
import torch.nn.functional as F
from transformers import AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
import multiprocessing as mp
from tqdm import tqdm
import re

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lmms_eval_adapter import load_task
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
    history: List[torch.Tensor]
    nfe: int
    L0: int

def build_llava_prompt(
    question: str,
    conv_template: str = "llava_llada",
    tokenizer_id: str | None = None,
    add_image_token: bool = True,
):
    import copy
    conv = copy.deepcopy(conv_templates[conv_template])
    if tokenizer_id:
        conv.tokenizer_id = tokenizer_id
    if add_image_token and DEFAULT_IMAGE_TOKEN not in question:
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

@torch.no_grad()
def generate_llada_v_with_history(
    model, tokenizer, input_ids, images, image_sizes,
    steps, gen_length, block_length, temperature,
    target, unmask_threshold=None, mask_id=126336,
    mcmc_alpha: float = 1.0, mcmc_steps: int = 0
) -> DiffusionOutput:
    position_ids = None
    attention_mask = None
    inputs_embeds = None
    (_input_ids, position_ids, attention_mask, _pkv, inputs_embeds, _labels) = model.prepare_inputs_labels_for_multimodal(
        input_ids, position_ids, attention_mask, None, None, images, ["image"], image_sizes=image_sizes
    )
    device = inputs_embeds.device
    B = inputs_embeds.shape[0]
    L0 = inputs_embeds.shape[1]
    d  = inputs_embeds.shape[2]
    assert B == 1
    total_len = L0 + gen_length
    masked_embed = model.get_model().embed_tokens(torch.tensor([mask_id], device=device))
    x_embeds = masked_embed.repeat(B, total_len, 1)
    x_embeds[:, :L0] = inputs_embeds
    x_tokens = torch.full((B, total_len), mask_id, dtype=torch.long, device=device)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (i < rem) for i in range(num_blocks)]
    nfe = 0
    hist: List[torch.Tensor] = []
    for blk in range(num_blocks):
        s = L0 + blk * block_length
        e = L0 + (blk + 1) * block_length
        cur_steps = steps_per_block[blk]

        def run_block_once(step_offset: int = 0, base_log_probs: torch.Tensor | None = None,
                           base_first_steps: torch.Tensor | None = None):
            if base_log_probs is None:
                token_log_probs = torch.full((B, block_length), float("nan"), device=device, dtype=torch.float64)
            else:
                token_log_probs = base_log_probs.clone()
            if base_first_steps is None:
                first_steps = torch.full((B, block_length), -1, device=device, dtype=torch.long)
            else:
                first_steps = base_first_steps.clone()

            time_steps_left = cur_steps - step_offset
            time_steps_left = max(1, time_steps_left)
            blk_tokens = x_tokens[:, s:e]
            eos_mask = (blk_tokens == 126348)
            if eos_mask.any():
                first_eos_pos = torch.where(eos_mask[0])[0][0].item()
                valid_region = torch.zeros_like(blk_tokens, dtype=torch.bool)
                valid_region[:, :first_eos_pos] = True
            else:
                valid_region = torch.ones_like(blk_tokens, dtype=torch.bool)
            current_mask_count = ((blk_tokens == mask_id) & valid_region).sum(dim=1).max().item()
            effective_steps = min(time_steps_left, max(int(current_mask_count), 1))
            num_transfer = get_num_transfer_tokens((blk_tokens == mask_id) & valid_region, effective_steps)

            outputs = model.get_model()(inputs_embeds=x_embeds, attention_mask=None, position_ids=None, use_cache=False, return_dict=True)
            logits = model.lm_head(outputs.last_hidden_state).float()
            mask_blk = (blk_tokens == mask_id) & valid_region
            x0_blk, tr_idx_blk, x0_p_blk = get_transfer_index(
                logits[:, s:e], temperature, target,
                mask_blk, blk_tokens, num_transfer[:, 0],
                threshold=unmask_threshold
            )
            if tr_idx_blk.any():
                x_slice = x_tokens[:, s:e]
                x_slice[tr_idx_blk] = x0_blk[tr_idx_blk]
                x_tokens[:, s:e] = x_slice
                x_embeds[:, s:e] = model.get_model().embed_tokens(x_tokens[:, s:e])
                update_mask = tr_idx_blk & (first_steps < 0)
                if update_mask.any():
                    token_log_probs[update_mask] = torch.log(
                        x0_p_blk[update_mask].clamp_min(1e-12)
                    ).to(token_log_probs.dtype)
                    first_steps[update_mask] = step_offset

            hist.append(x_tokens.clone().cpu())
            nfe_local = 1
            i_local = 1
            step_idx = step_offset + 1

            while True:
                nfe_local += 1
                step_transfer_idx = min(i_local, effective_steps - 1)
                blk_tokens = x_tokens[:, s:e]
                eos_mask = (blk_tokens == 126348)
                if eos_mask.any():
                    first_eos_pos = torch.where(eos_mask[0])[0][0].item()
                    valid_region = torch.zeros_like(blk_tokens, dtype=torch.bool)
                    valid_region[:, :first_eos_pos] = True
                else:
                    valid_region = torch.ones_like(blk_tokens, dtype=torch.bool)

                outputs = model.get_model()(inputs_embeds=x_embeds, attention_mask=None, position_ids=None, use_cache=False, return_dict=True)
                logits = model.lm_head(outputs.last_hidden_state).float()

                mask_blk = (x_tokens[:, s:] == mask_id)
                mask_blk[:, block_length:] = False
                mask_blk[:, :block_length] = mask_blk[:, :block_length] & valid_region
                x0, tr_idx, x0_p = get_transfer_index(
                    logits[:, s:], temperature, target,
                    mask_blk, x_tokens[:, s:], num_transfer[:, step_transfer_idx],
                    threshold=unmask_threshold
                )
                if tr_idx.any():
                    x_slice = x_tokens[:, s:]
                    x_slice[tr_idx] = x0[tr_idx]
                    x_tokens[:, s:] = x_slice
                    x0_embeds = model.get_model().embed_tokens(torch.where(tr_idx, x0, x_slice))
                    x_embeds[:, s:][tr_idx] = x0_embeds[tr_idx]
                    block_tr = tr_idx[:, :block_length]
                    update_mask = block_tr & (first_steps < 0)
                    if update_mask.any():
                        token_log_probs[update_mask] = torch.log(
                            x0_p[:, :block_length][update_mask].clamp_min(1e-12)
                        ).to(token_log_probs.dtype)
                        first_steps[update_mask] = step_idx

                hist.append(x_tokens.clone().cpu())
                remaining_masks = (x_tokens[:, s:e] == mask_id) & valid_region
                if remaining_masks.sum() == 0:
                    break
                i_local += 1
                step_idx += 1

            return x_tokens[:, s:e].clone(), token_log_probs, first_steps, nfe_local

        block_tokens, token_log_probs, first_steps, nfe_inc = run_block_once()
        nfe += nfe_inc
        current_tokens = block_tokens.clone()
        current_log_probs = token_log_probs.clone()
        current_first_steps = first_steps.clone()
        current_score = torch.nan_to_num(current_log_probs, nan=-1e9).sum(dim=1)

        for _ in range(mcmc_steps):
            rollback_t = random.randint(0, cur_steps - 1) if cur_steps > 1 else 0
            rollback_mask = (current_first_steps > rollback_t) & (current_first_steps >= 0)

            backup_tokens = current_tokens.clone()
            backup_log_probs = current_log_probs.clone()
            backup_first_steps = current_first_steps.clone()
            backup_score = current_score.clone()

            if rollback_mask.any():
                current_tokens[rollback_mask] = mask_id
                current_log_probs[rollback_mask] = float("nan")
                current_first_steps[rollback_mask] = -1
                x_tokens[:, s:e] = current_tokens
                x_embeds[:, s:e] = model.get_model().embed_tokens(x_tokens[:, s:e])

            block_tokens, token_log_probs, first_steps, nfe_inc = run_block_once(
                step_offset=rollback_t + 1,
                base_log_probs=current_log_probs,
                base_first_steps=current_first_steps,
            )
            nfe += nfe_inc
            new_tokens = block_tokens
            new_log_probs = token_log_probs
            new_first_steps = first_steps
            new_score = torch.nan_to_num(new_log_probs, nan=-1e9).sum(dim=1)

            ratio = (mcmc_alpha - 1.0) * (new_score - current_score)
            rand = torch.log(torch.rand_like(ratio).clamp_min(1e-12))
            accept_mask = (ratio >= 0) | (rand < ratio)

            for b in range(B):
                if accept_mask[b]:
                    current_tokens[b] = new_tokens[b]
                    current_log_probs[b] = new_log_probs[b]
                    current_first_steps[b] = new_first_steps[b]
                    current_score[b] = new_score[b]
                else:
                    current_tokens[b] = backup_tokens[b]
                    current_log_probs[b] = backup_log_probs[b]
                    current_first_steps[b] = backup_first_steps[b]
                    current_score[b] = backup_score[b]
                    x_tokens[b, s:e] = current_tokens[b]
            x_tokens[:, s:e] = current_tokens
            x_embeds[:, s:e] = model.get_model().embed_tokens(x_tokens[:, s:e])

        if (x_tokens[:, s:e] == 126348).any():
            break
    return DiffusionOutput(sequences=x_tokens, history=hist, nfe=nfe, L0=L0)

def get_transfer_index(logits, temperature, target, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    logits_with_noise[..., 126336] = -float('inf')
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
        return x0, selected, x0_p
    confidence = x0_p.masked_fill(~mask_index, float("-inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j].item() if torch.is_tensor(num_transfer_tokens[j]) else num_transfer_tokens[j])
        if k <= 0:
            continue
        _, sel = torch.topk(confidence[j], k=k)
        transfer_index[j, sel] = True
    return x0, transfer_index, x0_p

def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list

def get_prompt(data_i):
    return Template(system_prompts).render(problem=data_i["question"])

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

def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
    L = history[0].shape[1]
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)
    for t, snap in enumerate(history, start=1):
        cur = snap[sample_idx]
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
    unprocessed_mask = (step_map == 0)
    if unprocessed_mask.any():
        max_step = step_map.max()
        step_map[unprocessed_mask] = max_step
    return step_map

def worker(pretrained_model, rank, prompts, orig_idx, data_idx, image_paths, seq_dict, step_dict, imgabs_dict, batch_size, config):
    from PIL import Image
    import copy
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tokenizer_gpu, model_gpu, image_processor, max_length = load_pretrained_model(
        pretrained_model, None, "llava_llada", attn_implementation="sdpa", device_map=device
    )
    model_gpu.eval()
    special_tokens = {
        "additional_special_tokens": [DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    }
    num_new = tokenizer_gpu.add_special_tokens(special_tokens)
    if num_new > 0:
        try:
            model_gpu.resize_token_embeddings(len(tokenizer_gpu), mean_resizing=False)
        except TypeError:
            model_gpu.resize_token_embeddings(len(tokenizer_gpu))
    mask_id = 126336
    unmask_threshold = None if config.rollout.remasking_strategy == "low_confidence_static" else config.rollout.dynamic_threshold
    for j in tqdm(range(len(prompts)), desc=f"GPU {rank}", position=rank, leave=True):
        did = data_idx[j]
        prompt_text = prompts[j]
        image_payload = image_paths[j] if image_paths is not None else None
        if image_payload is not None:
            images = None
            if isinstance(image_payload, list):
                images = []
                for img in image_payload:
                    if img is None:
                        continue
                    if hasattr(img, "convert"):
                        images.append(img.convert("RGB"))
                        continue
                    if isinstance(img, dict) and img.get("path"):
                        src_path = img["path"]
                        images.append(Image.open(src_path).convert("RGB"))
                        continue
                    img_path = img
                    if not os.path.isabs(img_path) and hasattr(config.dataset, "image_root") and config.dataset.image_root is not None:
                        img_path = os.path.join(config.dataset.image_root, img_path)
                    img_path = os.path.abspath(img_path)
                    imgabs_dict[did] = img_path
                    images.append(Image.open(img_path).convert("RGB"))
            elif hasattr(image_payload, "convert"):
                images = [image_payload.convert("RGB")]
            elif isinstance(image_payload, dict) and image_payload.get("path"):
                src_path = image_payload["path"]
                images = [Image.open(src_path).convert("RGB")]
            else:
                img_path = image_payload
                if not os.path.isabs(img_path) and hasattr(config.dataset, "image_root") and config.dataset.image_root is not None:
                    img_path = os.path.join(config.dataset.image_root, img_path)
                img_path = os.path.abspath(img_path)
                imgabs_dict[did] = img_path
                images = [Image.open(img_path).convert("RGB")]

            image_tensor = process_images(images, image_processor, model_gpu.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
            image_sizes = [img.size for img in images]
        else:
            image_tensor = None
            image_sizes = None
        conv_prompt = build_llava_prompt(
            prompt_text,
            conv_template="llava_llada",
            tokenizer_id=pretrained_model,
            add_image_token=DEFAULT_IMAGE_TOKEN not in prompt_text,
        )
        input_ids = tokenizer_image_token(conv_prompt, tokenizer_gpu, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        out = generate_llada_v_with_history(
            model=model_gpu,
            tokenizer=tokenizer_gpu,
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            steps=config.rollout.steps,
            gen_length=config.rollout.max_gen_length,
            block_length=config.rollout.block_size,
            temperature=config.rollout.temperature,
            target=config.rollout.target,
            unmask_threshold=unmask_threshold,
            mask_id=mask_id,
            mcmc_alpha=getattr(config.rollout, "mcmc_alpha", 1.0),
            mcmc_steps=getattr(config.rollout, "mcmc_steps", 0),
        )
        out.sequences = out.sequences.cpu()
        L0 = model_gpu.prepare_inputs_labels_for_multimodal(input_ids, None, None, None, None, image_tensor, ["image"], image_sizes=image_sizes)[4].shape[1]
        seq_ids = out.sequences[:, L0:].tolist()
        text = tokenizer_gpu.batch_decode(seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0]
        m = denoise_step_map(out.history, mask_id=mask_id, sample_idx=0)
        step_map = m[L0:].tolist()
        seq_dict[orig_idx[j]] = text
        step_dict[orig_idx[j]] = step_map
    torch.cuda.empty_cache()

def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]

def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output

if __name__ == "__main__":
    config = get_config()
    mp.set_start_method("spawn", force=True)
    k_sample = config.rollout.num_response_per_task
    batch_size = config.rollout.batch_size
    project_name = config.experiment.project
    dataset = config.dataset.eval_dataset
    use_lmms_task = hasattr(config.dataset, "lmms_task") and config.dataset.lmms_task
    use_boxed = config.answer_must_in_box and not use_lmms_task
    if use_lmms_task:
        system_prompts = "{{problem}}"
    elif use_boxed:
        if "mmbench" in dataset.lower() or dataset.lower().startswith("mmb"):
            system_prompts = """<|startoftext|><|start_header_id|>user<|end_header_id|>This is a multiple-choice question. Answer with the option letter only (A, B, C, or D) in \\boxed{}. Do not include any other text.\n{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
        else:
            system_prompts = """<|startoftext|><|start_header_id|>user<|end_header_id|>You need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
    else:
        system_prompts = """<|startoftext|><|start_header_id|>user<|end_header_id|>{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
    pretrained_model = config.model
    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if use_lmms_task:
        task = load_task(
            config.dataset.lmms_task,
            tasks_dir=getattr(config.dataset, "lmms_task_dir", None),
            model_name=getattr(config.dataset, "lmms_model_name", None),
        )
        docs = list(task.eval_docs)
        doc_ids = list(range(len(docs)))
        if num_node > 1:
            doc_ids = get_data_chunk(doc_ids, num_node, node_index)
        num = len(doc_ids)
        data = []
    else:
        with open("../data/" + dataset + ".json", "r") as f:
            data = json.load(f)
        if num_node > 1:
            data = get_data_chunk(data, num_node, node_index)
        num = len(data)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        if use_lmms_task:
            doc_id = doc_ids[i]
            doc = docs[doc_id]
            prompt_text = task.doc_to_text(
                doc,
                lmms_eval_specific_kwargs=getattr(task, "lmms_eval_specific_kwargs", None),
            )
            data.append(
                {
                    "lmms_doc_id": doc_id,
                    "question": prompt_text,
                    "full_output": [],
                    "step_map": [],
                    "extracted_output": [],
                    "response_length": [],
                    "prompt": prompt_text,
                }
            )
        generation_prompts = generation_prompts + [get_prompt(data[i])] * k_sample
        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = get_prompt(data[i])
    image_paths = []
    for i in range(num):
        if use_lmms_task:
            image_paths += [task.doc_to_visual(docs[data[i]["lmms_doc_id"]]) if task.doc_to_visual else None] * k_sample
        else:
            image_paths += [data[i].get("image") or data[i].get("image_path")] * k_sample
    cprint("start generation...", "green")
    all_prompts = generation_prompts
    N = len(all_prompts)
    shuffled_idx = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]
    def split_even(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]
    n_gpu = torch.cuda.device_count()
    assert n_gpu > 1
    prompt_chunks = split_even(shuffled_prompts, n_gpu)
    idx_chunks = split_even(shuffled_idx, n_gpu)
    data_idx_full = [index_list[i] for i in shuffled_idx]
    data_idx_chunks = split_even(data_idx_full, n_gpu)
    image_paths_full = [image_paths[i] for i in shuffled_idx]
    image_chunks = split_even(image_paths_full, n_gpu)
    manager = mp.Manager()
    seq_dict = manager.dict()
    step_dict = manager.dict()
    imgabs_dict = manager.dict()
    procs = []
    for rk in range(n_gpu):
        p = mp.Process(target=worker,
                       args=(pretrained_model, rk,
                             prompt_chunks[rk],
                             idx_chunks[rk],
                             data_idx_chunks[rk],
                             image_chunks[rk],
                             seq_dict,
                             step_dict,
                             imgabs_dict,
                             batch_size,
                             config))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    restored_outputs = [seq_dict[i] for i in range(N)]
    restored_step_maps = [step_dict[i] for i in range(N)]
    for di in range(num):
        if di in imgabs_dict:
            data[di]["image_abs_path"] = imgabs_dict[di]
    def get_token_lengths(strings, tokenizer):
        pad_token = "<|mdm_mask|>"
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
    i = 0
    for full_output in restored_outputs:
        if use_boxed:
            extracted_output = extract_final_boxed_answer(full_output)
        else:
            if "</think>" in full_output:
                extracted_output = full_output.split("</think>")[1]
            else:
                extracted_output = full_output
        extracted_output = extracted_output.strip()
        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        data[index_i]["step_map"].append(restored_step_maps[i])
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])
        i += 1
    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    outputs_name = outputs_name + "-" + config.rollout.remasking_strategy
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
