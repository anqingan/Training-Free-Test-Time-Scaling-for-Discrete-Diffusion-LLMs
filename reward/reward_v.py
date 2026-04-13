import json
import os
import sys
from types import SimpleNamespace
import math_utils_v
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint
from typing import List, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lmms_eval_adapter import load_task
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def remove_endoftext(obj: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Remove all occurrences of '<|endoftext|>' from a string or from each string in a list.
    Non-str elements in the list are left unchanged.
    """
    token = '<|endoftext|>'
    if isinstance(obj, str):
        return obj.replace(token, '')
    if isinstance(obj, list):
        return [s.replace(token, '') if isinstance(s, str) else s for s in obj]
    raise TypeError("Expected str or list of str")


def run_lmms_eval(config, data, outputs_result_name):
    task_name = config.dataset.lmms_task
    task = load_task(
        task_name,
        tasks_dir=getattr(config.dataset, "lmms_task_dir", None),
        model_name=getattr(config.dataset, "lmms_model_name", None),
    )
    if "generate_until" not in task.get_config("output_type"):
        raise ValueError(f"Unsupported output_type for lmms-eval alignment: {task.get_config('output_type')}")

    from lmms_eval.evaluator_utils import TaskOutput

    task.args = SimpleNamespace(output_path=os.path.abspath(os.path.join("..", config.experiment.project)))
    task_output = TaskOutput.from_taskdict(task_name, task)
    task_output.task = task
    task_output.task_name = task_name
    task_output.args = task.args

    if getattr(config.rollout, "num_response_per_task", 1) > 1:
        cprint("lmms-eval alignment uses the first response per task.", color="yellow")

    filter_key = "none"
    for i, item in enumerate(data):
        doc_id = item.get("lmms_doc_id", i)
        doc = task.eval_docs[doc_id]
        preds = item.get("extracted_output", [])
        pred = preds[0] if preds else ""
        metrics = task.process_results(doc, [pred])
        for metric, value in metrics.items():
            task_output.sample_metrics[(metric, filter_key)].append(value)

    task_output.calculate_aggregate_metric(bootstrap_iters=0)
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "w") as f:
        for (metric, _filter), value in task_output.agg_metrics.items():
            f.write(f"{metric}: {value}\n")


if __name__ == "__main__":

    config = get_config()

    project_name = config.experiment.project
    
    
    

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model

    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    outputs_name = outputs_name + "-" + config.rollout.remasking_strategy
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, "r") as f:
        data = json.load(f)

    use_lmms_task = hasattr(config.dataset, "lmms_task") and config.dataset.lmms_task
    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    if use_lmms_task:
        run_lmms_eval(config, data, outputs_result_name)
        sys.exit(0)


    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    response_length_list = []
    for i in range(len(data)):
        
        response_length_list = response_length_list + data[i]["response_length"]
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        extracted_output_list = extracted_output_list + data[i]["extracted_output"]
        if config.dataset.data_type == "math" or config.dataset.data_type == "mmu":
            data[i]["correctness"] = []
            ground_truth_list = ground_truth_list + [data[i]["ground_truth_answer"]] * len(data[i]["extracted_output"])

    

    if config.dataset.data_type == "math" or config.dataset.data_type == "mmu":

        nest_asyncio.apply()

        async def get_correctness():
            executor = ThreadPoolExecutor(max_workers=64)
            tasks = []
            for i in range(len(index_list)):
                tasks.append(math_utils_v.is_equal(remove_endoftext(extracted_output_list[i]), ground_truth_list[i], executor))
            results = await asyncio.gather(*tasks)
            return results
    
        correctness_list = asyncio.run(get_correctness())
        for i in range(len(index_list)):
            index_i = index_list[i]
            data[index_i]["correctness"].append(correctness_list[i])



    def z_score_normalize(lst):
        mean = sum(lst) / len(lst)
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        if std == 0:
            return [0 for x in lst]
        return [(x - mean) / std for x in lst]






    def set_last_t(lst: list, t: int) -> None:
        new_lst = lst.copy()
        new_val = max(lst) + 1
        new_lst[-t:] = [new_val] * t
        return new_lst



    if config.dataset.data_type == "math" or config.dataset.data_type == "mmu":
        acc = sum(correctness_list)/len(correctness_list)
    else:
        num_task   = 0
        num_correct_task = 0
        for x in data:
            for y in x["correctness"]:
                num_correct_task += all(y)
                num_task += 1
        acc = num_correct_task / num_task if num_task else 0

    if config.rollout.output_unmasking_history == False:
        for i in range(len(data)):
            data[i]["step_map"] = []
    
    import os
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        
        
        avg_len = sum(response_length_list)/len(response_length_list)

        save_and_print(f"acc: {acc}\navg length: {avg_len}")
