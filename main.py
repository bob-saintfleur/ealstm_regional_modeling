"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import torch
import torch.nn as nn
import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
from selectbv import select_bv_by_class
from my_multiproc import run_parallel, dispatch_args_to_cfg
import glob
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from papercode.datasets import CamelsH5, CamelsTXT, GetClimSubset
from papercode.datautils import (add_camels_attributes, load_attributes,
                                 rescale_features)
from papercode.ealstm import EALSTM
from papercode.lstm import LSTM
from papercode.metrics import calc_nse
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files, get_basin_list

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 320,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 270,
    'train_start': pd.to_datetime('01101996', format='%d%m%Y'),  # ex 1999
    'train_end': pd.to_datetime('30092007', format='%d%m%Y'),  # 2008
    'val_start': pd.to_datetime('01091996', format='%d%m%Y'),  # 1989
    'val_end': pd.to_datetime('30091996', format='%d%m%Y'),  # 1999
    'test_start': pd.to_datetime('01102007', format='%d%m%Y'),
    'test_end': pd.to_datetime('30092008', format='%d%m%Y')
}

# check if GPU is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############
# Prepare run #
###############


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "climatology", "evaluate_test", 'hindcast',
                                         "eval_robustness"])
    parser.add_argument('--camels_root', type=str, help="Root directory of CAMELS data set")
    parser.add_argument('--device', type=str, default="cpu", help="Device to target")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--hp', type=int, required=False, help="Forecasting lead time")
    parser.add_argument('--nproc_bv', type=int, default=1, required=False, help="Subdivide nbv into tasks")
    parser.add_argument('--basins_file', type=str, required=False, help="a specific file of basins")
    parser.add_argument('--list_bv', nargs="+", required=False, help="a specific list of basins")
    parser.add_argument('--nbv', type=int, default=None, required=False,
                        help="Select uniformly n basins according to their NSE rank in Kratzert et al. (2019)")
    parser.add_argument('--ref_period_clim', nargs="+", default=("20070801", "20080820"), required=False,
                        metavar=("yyyymmdd", "yyyymmdd"), help="Reference one/two years period for evaluation")
    parser.add_argument('--clim_years', nargs="+", default=(1989, 2008), required=False,
                        metavar=("lower_year", "upper_year"), help="Limit years for climatology")
    parser.add_argument('--global', type=bool, default=False, help="Eval on full period")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument('--models_box', type=str, help="For evaluation mode. Path to run directory lot.")
    parser.add_argument('--cache_data',
                        type=bool,
                        default=False,
                        help="If True, loads all data into memory")
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help="Number of parallel threads for data loading")
    parser.add_argument('--no_static',
                        type=bool,
                        default=False,
                        help="If True, trains LSTM without static features")
    parser.add_argument('--concat_static',
                        type=bool,
                        default=False,
                        help="If True, train LSTM with static feats concatenated at each time step")
    parser.add_argument('--use_mse',
                        type=bool,
                        default=False,
                        help="If True, uses mean squared error as loss function.")
    cfg = vars(parser.parse_args())
    if cfg["basins_file"] is not None:
        list_bv = [a.split()[0] for a in open(str(Path(cfg["camels_root"]).parent)+"/"+cfg["basins_file"]).readlines()]
        cfg.update({"list_bv":list_bv})

    # Validation checks
    if (cfg["mode"] == "train") and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if cfg["mode"] != "train" and cfg["run_dir"] is None and cfg["models_box"] is None:
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    # I have added the following to consider a global evaluation
    if (cfg["global"] is True) and (cfg["mode"] != "train"):
        cfg["test_start"] = pd.to_datetime('01011980', format='%d%m%Y')
        cfg["test_end"] = pd.to_datetime('31122014', format='%d%m%Y')
        GLOBAL_SETTINGS["test_start"] = pd.to_datetime('01011980', format='%d%m%Y')
        GLOBAL_SETTINGS["test_end"] = pd.to_datetime('31122014', format='%d%m%Y')

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run
    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    run_name = f'run_{day}{month}_{hour}{minute}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / run_name
    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
        cfg["val_dir"].mkdir(parents=True)

        cfg["test_dir"] = cfg["run_dir"] / 'data' / 'test'
        cfg["test_dir"].mkdir(parents=True)
    else:
        raise RuntimeError(f"There is already a folder at {cfg['run_dir']}")

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, Path):  # i replaced PosixPath by Path
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    basins : List
        List containing the 8-digit USGS gauge id

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(camels_root=cfg["camels_root"],
                    out_file=cfg["train_file"],
                    basins=basins,
                    dates=[cfg["train_start"], cfg["train_end"]],
                    with_basin_str=True,
                    seq_length=cfg["seq_length"])
    return cfg


################
# Define Model #
################


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connected layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        if self.concat_static or self.no_static:
            self.lstm = LSTM(input_size=input_size_dyn,
                             hidden_size=hidden_size,
                             initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None

        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        return out, h_n, c_n


###########################
# Train or evaluate model #
###########################


def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    basins = get_basin_list()

    # create folder structure for this run
    cfg = _setup_run(cfg)

    # prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)

    # prepare PyTorch DataLoader
    ds = CamelsH5(h5_file=cfg["train_file"],
                  basins=basins,
                  db_path=cfg["db_path"],
                  concat_static=cfg["concat_static"],
                  cache=cfg["cache_data"],
                  no_static=cfg["no_static"])
    loader = DataLoader(ds,
                        batch_size=cfg["batch_size"],
                        shuffle=True,
                        num_workers=cfg["num_workers"])

    # create model and optimizer
    input_size_stat = 0 if cfg["no_static"] else 27
    input_size_dyn = 5 if (cfg["no_static"] or not cfg["concat_static"]) else 32
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=cfg["hidden_size"],
                  initial_forget_bias=cfg["initial_forget_gate_bias"],
                  dropout=cfg["dropout"],
                  concat_static=cfg["concat_static"],
                  no_static=cfg["no_static"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # define loss function
    if cfg["use_mse"]:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss()

    # reduce learning rates after each 10 epochs
    learning_rates = {11: 5e-4, 21: 1e-4}

    for epoch in range(1, cfg["epochs"] + 1):
        # set new learning rate
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        train_epoch(model, optimizer, loss_func, loader, cfg, epoch, cfg["use_mse"])

        model_path = cfg["run_dir"] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int, use_mse: bool):
    """Train model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    optimizer : torch.optim.Optimizer
        Optimizer used for weight updating
    loss_func : nn.Module
        The loss function, implemented as a PyTorch Module
    loader : DataLoader
        PyTorch DataLoader containing the training data in batches.
    cfg : Dict
        Dictionary containing the run config
    epoch : int
        Current Number of epoch
    use_mse : bool
        If True, loss_func is nn.MSELoss(), else NSELoss() which expects additional std of discharge
        vector

    """
    model.train()

    # process bar handle
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    # Iterate in batches over training set
    for data in pbar:
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        if len(data) == 3:
            x, y, q_stds = data
            x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
            predictions = model(x)[0]

        # forward pass through EALSTM
        elif len(data) == 4:
            x_d, x_s, y, q_stds = data
            x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
            predictions = model(x_d, x_s[:, 0, :])[0]

        # MSELoss
        if use_mse:
            loss = loss_func(predictions, y)

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(DEVICE)
            loss = loss_func(predictions, y, q_stds)

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")


def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config

    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    basins = get_basin_list()
    basins.sort()  # FIXME: added to control sth on July18th
    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path,
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # create model
    input_size_stat = 0 if run_cfg["no_static"] else 27
    input_size_dyn = 5 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else 32
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"],
                  concat_static=run_cfg["concat_static"],
                  no_static=run_cfg["no_static"]).to(DEVICE)

    # load trained model
    weight_file = user_cfg["run_dir"] / 'model_epoch30.pt'
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
    results = {}
    for basin in tqdm(basins):
        ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
                            basin=basin,
                            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                            is_train=False,
                            seq_length=run_cfg["seq_length"],
                            with_attributes=True,
                            attribute_means=means,
                            attribute_stds=stds,
                            concat_static=run_cfg["concat_static"],
                            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)
        preds, obs = evaluate_basin(model, loader)

        df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)

        results[basin] = df

    _store_results(user_cfg, run_cfg, results)


def make_ref_dates(ref_date: tuple, format_="%Y%m%d"):
    list_ref_date = pd.date_range(start=pd.to_datetime(ref_date[0], format=format_),
                                  end=pd.to_datetime(ref_date[1], format=format_), freq='D')
    list_ref_date = [a.strftime("%Y%m%d") for a in list_ref_date if a.strftime("%m%d") != "0229"]
    list_ref_date.sort()
    return list_ref_date


def ens_run(user_cfg: Dict):
    """Launch climatology or hindcast mode using options such as the basin, the evaluation period, the models
    dir, the lead times.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    """
    with open(rf'{user_cfg["run_dir"]}/cfg.json', 'r') as fp:
        run_cfg = json.load(fp)
    basins = get_basin_list()
    basins.sort()

    # # get attribute means/stds
    db_path = str(user_cfg["run_dir"]) + "/attributes.db"
    attributes = load_attributes(db_path=db_path,
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # create model
    input_size_stat = 0 if run_cfg["no_static"] else 27
    input_size_dyn = 5 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else 32
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"],
                  concat_static=run_cfg["concat_static"],
                  no_static=run_cfg["no_static"]).to(DEVICE)

    # load trained model
    weight_file = str(user_cfg["run_dir"]) + '/model_epoch30.pt'
    model.load_state_dict(torch.load(weight_file, weights_only=True, map_location=DEVICE))

    results = {}
    run_mode = user_cfg["mode"]
    discr_mode = "hcst_" if "hind" in run_mode else 'clim_'

    if user_cfg["list_bv"]:
        basins_l = user_cfg["list_bv"]
    elif user_cfg["nbv"] is not None:
        # basins_l = get_sub_bv_uniformly(size=user_cfg["nbv"])
        basins_l = select_bv_by_class(size=user_cfg["nbv"])
    else:
        basins_l = basins
    # for basin in tqdm(basins):
    for basin in basins_l:
        period = user_cfg["ref_period_clim"]
        all_mbr_date = GetClimSubset(camels_root=user_cfg["camels_root"],
                                     basin=basin,
                                     dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                                     is_train=False,
                                     is_clim=True,
                                     ref_date=period[0],
                                     hp=user_cfg["hp"],
                                     seq_length=run_cfg["seq_length"],
                                     with_attributes=True,
                                     attribute_means=means,
                                     attribute_stds=stds,
                                     concat_static=run_cfg["concat_static"],
                                     period=period, run_mode=run_mode,
                                     db_path=db_path).get_clim_dates()

        out_c, ref_dt = (), ()
        BAR_CLIM = tqdm(all_mbr_date, desc=f"{run_mode.upper()} - b: {basin}: It-Ens-Mbr-Dt", leave=False)
        for dte, f_dt, all_clim in BAR_CLIM:
            BAR_CLIM.set_postfix_str(f"Date: {dte}")

            # Every scenario is considered in one shot
            all_clim = dict(sorted(all_clim.items()))
            ds_x = torch.cat([a[0] for a in all_clim.values()], dim=0)
            ds_y = torch.cat([a[1] for a in all_clim.values()], dim=0)
            ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
                                basin=basin,
                                dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                                is_train=False,
                                seq_length=run_cfg["seq_length"],
                                with_attributes=True,
                                attribute_means=means,
                                attribute_stds=stds,
                                concat_static=run_cfg["concat_static"],
                                db_path=db_path,
                                preset_xy_tensor=(ds_x, ds_y))

            loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False, num_workers=1)
            pred, obs = evaluate_basin(model, loader)
            df = pd.DataFrame(np.concatenate([obs[-1:][np.newaxis], pred], axis=0).T, index=[f_dt],
                              columns=["y_obs"] + list(all_clim.keys()))
            df.index.name = "Date"
            out_c += (df,)
        out_c = pd.concat(out_c, axis=0).sort_index()
        results[basin] = out_c
    if user_cfg["nproc_bv"] is not None:
        return user_cfg, run_cfg, results, f'{discr_mode}hp{user_cfg["hp"]}'
    else:
        _store_results(user_cfg, run_cfg, results, f'{discr_mode}hp{user_cfg["hp"]}')


def climatology(user_cfg: Dict):
    """Run climatology"""
    if user_cfg["mode"] == "climatology":
        return ens_run(user_cfg=user_cfg)


def hindcast(user_cfg: Dict):
    """Run hindcast"""
    if user_cfg["mode"] == "hindcast":
        return ens_run(user_cfg=user_cfg)


def evaluate_test(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config

    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    basins = get_basin_list()

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path,
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # create model
    input_size_stat = 0 if run_cfg["no_static"] else 27
    input_size_dyn = 5 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else 32
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"],
                  concat_static=run_cfg["concat_static"],
                  no_static=run_cfg["no_static"]).to(DEVICE)

    # load trained model
    weight_file = user_cfg["run_dir"] / 'model_epoch30.pt'
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    date_range = pd.date_range(start=GLOBAL_SETTINGS["test_start"], end=GLOBAL_SETTINGS["test_end"])
    results = {}
    for basin in tqdm(basins):
        ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
                            basin=basin,
                            dates=[GLOBAL_SETTINGS["test_start"], GLOBAL_SETTINGS["test_end"]],
                            is_train=False,
                            seq_length=run_cfg["seq_length"],
                            with_attributes=True,
                            attribute_means=means,
                            attribute_stds=stds,
                            concat_static=run_cfg["concat_static"],
                            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)
        preds, obs = evaluate_basin(model, loader)
        df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)
        results[basin] = df

    # _store_results(user_cfg, run_cfg, results)
    if user_cfg["nproc_bv"]:
        return user_cfg, run_cfg, results
    else:
        _store_results(user_cfg, run_cfg, results)


def evaluate_basin(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()
    preds, obs = None, None
    with torch.no_grad():
        for data in loader:
            if len(data) == 2:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = model(x)[0]
            elif len(data) == 3:
                x_d, x_s, y = data
                x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
                p = model(x_d, x_s[:, 0, :])[0]

            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)
        preds = rescale_features(preds.numpy(), variable='output')
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0
    return preds, obs


def eval_robustness(user_cfg: Dict):
    """Evaluate model robustness of EA-LSTM

    In this experiment, gaussian noise with increasing scale is added to the static features to
    evaluate the model robustness against pertubations of the static catchment characteristics.
    For each scale, 50 noise vectors are drawn.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config

    Raises
    ------
    NotImplementedError
        If the run_dir specified points not to a EA-LSTM model folder.
    """
    random.seed(user_cfg["seed"])
    np.random.seed(user_cfg["seed"])

    # fixed settings for this analysis
    n_repetitions = 50
    scales = [0.1 * i for i in range(11)]

    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    if run_cfg["concat_static"] or run_cfg["no_static"]:
        raise NotImplementedError("This function is only implemented for EA-LSTM models")

    basins = get_basin_list()

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path,
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # initialize Model
    model = Model(input_size_dyn=5,
                  input_size_stat=27,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"]).to(DEVICE)
    weight_file = user_cfg["run_dir"] / "model_epoch30.pt"
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    overall_results = {}
    # process bar handle
    pbar = tqdm(basins, file=sys.stdout)
    for basin in pbar:
        ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
                            basin=basin,
                            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                            is_train=False,
                            with_attributes=True,
                            attribute_means=means,
                            attribute_stds=stds,
                            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False, num_workers=0)
        basin_results = defaultdict(list)
        step = 1
        for scale in scales:
            for _ in range(1 if scale == 0.0 else n_repetitions):
                noise = np.random.normal(loc=0, scale=scale, size=27).astype(np.float32)
                noise = torch.from_numpy(noise).to(DEVICE)
                nse = eval_with_added_noise(model, loader, noise)
                basin_results[scale].append(nse)
                pbar.set_postfix_str(f"Basin progress: {step}/{(len(scales) - 1) * n_repetitions + 1}")
                step += 1

        overall_results[basin] = basin_results
    out_file = (Path(__file__).absolute().parent /
                f'results/{user_cfg["run_dir"].name}_model_robustness.p')
    if not out_file.parent.is_dir():
        out_file.parent.mkdir(parents=True)
    with out_file.open("wb") as fp:
        pickle.dump(overall_results, fp)


def eval_with_added_noise(model: torch.nn.Module, loader: DataLoader, noise: torch.Tensor) -> float:
    """Evaluate model on a single basin with added noise

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.
    noise : torch.Tensor
        Tensor containing the noise for this evaluation run.

    Returns
    -------
    float
        Nash-Sutcliff-Efficiency of the simulations with added noise.
    """
    model.eval()
    preds, obs = None, None
    with torch.no_grad():
        for x_d, x_s, y in loader:
            x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
            batch_noise = noise.repeat(*x_s.size()[:2], 1)
            x_s = x_s.add(batch_noise)
            y_hat = model(x_d, x_s[:, 0, :])[0]

            if preds is None:
                preds = y_hat.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, y_hat.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        obs = obs.numpy()
        preds = rescale_features(preds.numpy(), variable='output')

        # set discharges < 0 to zero
        preds[preds < 0] = 0

        nse = calc_nse(obs[obs >= 0], preds[obs >= 0])
        return nse


def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame or Dict, discr: str = None):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : Dict
        Dictionary containing the predicted discharge with basins as keys.
    """
    discr_ = f"_{discr}" if discr is not None else ""
    if run_cfg["no_static"]:
        file_name = str(user_cfg["run_dir"]) + f"/lstm_no_static_seed{run_cfg['seed']}{discr_}.p"
    else:
        if run_cfg["concat_static"]:
            file_name = str(user_cfg["run_dir"]) + f"/lstm_seed{run_cfg['seed']}{discr_}.p"
        else:
            file_name = str(user_cfg["run_dir"]) + f"/ealstm_seed{run_cfg['seed']}{discr_}.p"
    with open(file_name, 'wb') as fp:
        pickle.dump(results, fp)
    print(f"Successfully store results at {file_name}")


def get_basin_list_by_args(u_cfg: Dict):
    """

    Parameters
    ----------
    u_cfg

    Returns
    -------

    """
    if u_cfg["list_bv"]:
        basins = u_cfg["list_bv"]
    elif u_cfg["nbv"] is not None:
        basins = select_bv_by_class(size=u_cfg["nbv"])
    else:
        basins = get_basin_list()
    return basins


if __name__ == "__main__":
    # config = get_args()
    # globals()[config["mode"]](config)
    config_0 = get_args()
    list_md_cfg = ()
    if config_0["models_box"]: # use a box of run_dir (or models)
        for r_d in glob.glob(rf'{config_0["models_box"]}/run_*seed*'):
            config_x = config_0.copy()
            config_x["run_dir"] = r_d
            list_md_cfg += (config_x,)
    else:
        list_md_cfg = [config_0]
    for config_z in list_md_cfg:
        if config_z["nproc_bv"] is not None:
            basins_l = get_basin_list_by_args(config_z)
            list_cfg = dispatch_args_to_cfg(config_z, "list_bv", basins_l, "nproc_bv")
        else:
            list_cfg = [config_z]
        run_parallel(globals()[config_z["mode"]], list_cfg)
