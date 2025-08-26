# !/bin/env python3

import multiprocessing as mp
import os
from typing import Dict, List, Tuple
import pickle
import queue
import time
from glob import glob
import pandas as pd
# import psutil

SENTINEL = None


def do_work(pending_task, completed_task):
    """ use args and function and run as task while controling the flow"""
    # Get the current workers' name
    worker_name = mp.current_process().name
    worker_task = None
    while True:
        try:
            task = pending_task.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
        else:
            try:
                if task == SENTINEL:
                    break
                time_start = time.perf_counter()
                worker_task = pickle.loads(task["func"])
                result = worker_task(**task["task"])
                completed_task.put({worker_task.__name__: result})
                time_end = time.perf_counter() - time_start
                print(f" {worker_name} finished: working time = {round(time_end)} s")

            except Exception as e:
                print(f"{worker_name} task failed {str(e)}")
                completed_task.put({worker_task.__name__: None})


def par_proc(job_list, num_cpus=None):
    """ Perform a parallel processing of running task using a list of task"""
    # Get the number of cores

    if not num_cpus:
        num_cpus = mp.cpu_count() - 4
        # num_cpus = psutil.cpu_count(logical=False)
    # set up queues and I/O files from servers
    pending_task = mp.Queue()
    completed_task = mp.Queue()

    processes, results = [], []

    # task pointer
    num_tasks = 0
    for job in job_list:
        for task in job["tasks"]:
            exp_jobs = {}
            num_tasks += 1
            exp_jobs.update({'func': pickle.dumps(job['func'])})
            exp_jobs.update({'task': task})
            pending_task.put(exp_jobs)

    num_workers = num_cpus
    for c in range(num_workers):
        pending_task.put(SENTINEL)
    for c in range(num_workers):
        p = mp.Process(target=do_work, args=(pending_task, completed_task))
        p.name = f'worker{c}'
        processes.append(p)
        p.start()

    completed_task_counter = 0
    while completed_task_counter < num_tasks:
        results.append(completed_task.get())
        completed_task_counter += 1
    for p in processes:
        p.join()
    return results


def dispatch_args_to_cfg(cfg: Dict, arg_to_update: str, list_to_split: List,
                         arg_sizer: str = "nproc_bv") -> List[Dict]:
    """
    Map an update on a parser returning a list of child from that parser.

    Parameters
    ----------
        cfg : the parent parser
        arg_to_update : the last key to be update
        list_to_split : the list generating the child specificities
        arg_sizer : the parameter which holds the number of children to have

    Returns
    -------
        A list of children
    """
    list_dict = []
    if cfg[arg_sizer]:
        if len(list_to_split) <= cfg[arg_sizer]:
            cfg[arg_sizer] = len(list_to_split)
        if len(list_to_split) >= cfg[arg_sizer]:
            r_sub = len(list_to_split) % cfg[arg_sizer]
            by_sub = len(list_to_split) // cfg[arg_sizer] + (0 if r_sub == 0 else 1)
            list_sub_ = [list_to_split[i:i + by_sub] for i in range(0, len(list_to_split), by_sub)]
        else:
            list_sub_ = [list_to_split]
        for sub_l in list_sub_:
            cfg[arg_to_update] = sub_l
            temp_dct = cfg.copy()
            list_dict.append(temp_dct)
            del temp_dct
    return list_dict


def store_multiproc_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame or Dict, discr: str = None):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results :
        Dict or DataFrame containing the observed and predicted discharge.
    discr : str to indentify the context of the model

    """
    discr_ = f"_{discr}" if discr is not None else ""

    if run_cfg["no_static"]:
        file_name = user_cfg["run_dir"] + f"/lstm_no_static_seed{run_cfg['seed']}{discr_}.p"
    else:
        if run_cfg["concat_static"]:
            file_name = user_cfg["run_dir"] + f"/lstm_seed{run_cfg['seed']}{discr_}.p"
        else:
            file_name = user_cfg["run_dir"] + f"/ealstm_seed{run_cfg['seed']}{discr_}.p"

    with open(file_name, 'wb') as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully store results at {file_name}")


def run_parallel(func_, list_cfg: List[Dict]):
    """ Run func_ separately on dates. The dates are divided and ran separately on replicated config"""
    list_task = [{"func": func_, "tasks": [dict(user_cfg=cfg_x) for cfg_x in list_cfg]}]
    results = par_proc(list_task)

    # get, bind and save results to specified path
    mode_ = list(results[0].keys())[0]
    user_cfg, run_cfg, _, discr_ = results[0][mode_]
    all_results = {}
    for sub_ in range(len(results)):
        res = results[sub_][mode_][2]
        all_results.update(res)
    store_multiproc_results(user_cfg, run_cfg, all_results, discr_)



