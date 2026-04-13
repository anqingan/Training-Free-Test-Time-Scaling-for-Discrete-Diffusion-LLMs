import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor

import math_utils
import nest_asyncio
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":

    config = get_config()

    project_name = config.experiment.project
    
    
    

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model

    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        data = json.load(f)


    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    question_list = []
    response_length_list = []
    for i in range(len(data)):
        
        response_length_list = response_length_list + data[i]["response_length"]
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        extracted_output_list = extracted_output_list + data[i]["extracted_output"]
        if config.dataset.data_type == "math":
            data[i]["correctness"] = []
            ground_truth_list = ground_truth_list + [data[i]["ground_truth_answer"]] * len(data[i]["extracted_output"])
            question_list = question_list + [data[i]["question"]] * len(data[i]["extracted_output"])

    

    is_sudoku = "sudoku" in str(config.dataset.eval_dataset).lower()

    if config.dataset.data_type == "math" and not is_sudoku:

        nest_asyncio.apply()

        async def get_correctness():
            executor = ThreadPoolExecutor(max_workers=64)
            tasks = []
            for i in range(len(index_list)):
                tasks.append(math_utils.is_equal(extracted_output_list[i], ground_truth_list[i], executor))
            results = await asyncio.gather(*tasks)
            return results
    
        correctness_list = asyncio.run(get_correctness())
        for i in range(len(index_list)):
            index_i = index_list[i]
            data[index_i]["correctness"].append(correctness_list[i])

    if config.dataset.data_type == "math" and is_sudoku:
        def extract_puzzle(question: str) -> str:
            if len(question) == 16 and question.isdigit():
                return question
            match = re.search(r"Puzzle:\s*([0-9]{16})", question)
            if match:
                return match.group(1)
            match = re.search(r"Sudoku puzzle:\s*([0-9]{16})", question)
            if match:
                return match.group(1)
            digits = re.findall(r"[0-9]", question)
            return "".join(digits[:16])

        def normalize_solution(solution_str: str) -> str:
            if solution_str is None:
                solution_str = ""
            match = re.search(r"([1-4]{16})", solution_str)
            if match:
                return match.group(1)
            digits = re.findall(r"[1-4]", solution_str)
            solution_str = "".join(digits)
            if len(solution_str) < 16:
                solution_str = solution_str + ("0" * (16 - len(solution_str)))
            elif len(solution_str) > 16:
                solution_str = solution_str[:16]
            return solution_str

        correctness_list = []
        for i in range(len(index_list)):
            question = question_list[i]
            puzzle_str = extract_puzzle(question)
            solution_str = normalize_solution(extracted_output_list[i])
            ground_truth = ground_truth_list[i]

            empty_indices = [j for j in range(16) if puzzle_str[j] == "0"]
            empty_cells = len(empty_indices)
            if empty_cells == 0:
                accuracy = 1.0 if solution_str == ground_truth else 0.0
            else:
                correct_cells = sum(
                    1 for j in empty_indices if solution_str[j] == ground_truth[j]
                )
                accuracy = correct_cells / empty_cells
            correctness_list.append(accuracy)
            data[index_list[i]]["correctness"].append(accuracy)



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


    if config.dataset.data_type == "math":
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
