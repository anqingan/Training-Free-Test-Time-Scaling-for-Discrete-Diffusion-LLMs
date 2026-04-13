import os
import sys
from typing import Optional


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def default_lmms_eval_root() -> str:
    return os.path.join(_repo_root(), "LLaDA-V-upstream", "eval", "lmms-eval")


def default_tasks_dir() -> str:
    return os.path.join(default_lmms_eval_root(), "lmms_eval", "tasks")


def ensure_lmms_eval_on_path(lmms_eval_root: Optional[str] = None) -> str:
    root = lmms_eval_root or default_lmms_eval_root()
    if not os.path.isdir(root):
        raise FileNotFoundError(f"lmms-eval root not found: {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def load_task(task_name: str, tasks_dir: Optional[str] = None, model_name: Optional[str] = None):
    ensure_lmms_eval_on_path()
    from lmms_eval.tasks import TaskManager, get_task_dict

    task_manager = TaskManager(include_path=tasks_dir or default_tasks_dir(), model_name=model_name)
    task_dict = get_task_dict([task_name], task_manager)
    if task_name not in task_dict:
        raise ValueError(f"Task not found: {task_name}")
    task_obj = task_dict[task_name]
    if isinstance(task_obj, tuple):
        _, task_obj = task_obj
    if isinstance(task_obj, dict):
        raise ValueError(f"Task '{task_name}' is a group; use a subtask name.")
    return task_obj
