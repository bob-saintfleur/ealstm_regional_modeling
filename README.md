# NOTE on this Forked project from Kratert et al. (2019)
This is a fork of the [ealstm models](git@github.com:kratzert/ealstm_regional_modeling.git) of Kratzert et al. (2019) 
which was published alongside this [paper](https://www.hydrol-earth-syst-sci.net/23/5089/2019/hess-23-5089-2019.html). 

The models were used in the present work only under an evaluation mode. These evaluation modes are declined as "hindcast"
and "climatology", and ensemble-based. The related findings are tied to above paper submitted recently for reviewing. 
The few lines that was added in this original code, in order to get it adapted to the needs of our experiments can be 
highlighted through versions comparison. The indications to re-run our experiments are provided right below. Note that 
for any usage of this LSTM code, credits remain belong first to Kratzert et al. (2019).

If you are discovering the original ealstm code, please jump to the original instructions below the *End of our changes* 
section (See below), or checkout the original paper first, then come back. Before you jump into our changes, make sure 
the CAMELS data as required in the original paper, including the runs provided are downloaded on your system, since our
experiments are based upon them.

## Starting point of our changes
The considered evaluation period used for the above paper ranges from 1989-10-01 to 1991-09-30 in a yyyy-mm-dd format.
The lead times range from 1 to 7 days.
The hindcast ensemble-based evaluation requires hindcast archives for each basin within the concerned period, including 
the covered lead times. The related data are in the **hindcast** sub folder provided in the data repo.
The climatology ensemble-based evaluation is based on the forcing records provided in the CAMELS dataset itself.

## 1. Main command line structure 
```
python main.py mode --camels_root path/to/CAMELS --models_box path/to/runs --basins_file name_of_basins_list_file --hp value --ref_period_clim start_date end_date
```
This will perform evaluation on the models found in **path\to\runs** for the **mode**. It requires a positive lead time 
(e.g. --hp 1), the period to evaluate (e.g. --ref_period_clim 19901001 19901030 ). Data are expected from *path/to/CAMELS* 
as it will found sub-folders such as _basin_mean_forcing/_, _usgs_streamflow/_, _camels_attributes_v2.0/_ , as disposed 
in the original CAMELS dataset structure. The number of basins will to be used is passed using the basins_file argument
and a file_name.

Note : Parallelization on sub-periods can be done using * --nproc_bv value * where value is an integer between 2 and 20

## 2. Example
You may need to clone this the code first

````
git clone https://github.com/bob-saintfleur/ealstm_regional_modeling.git -b hydro_uge
````

You may also need to set up a proper environment if you don't intend to use uv, see original paper steps for environment setting

### 2.1.  Launch with uv

- Climatology
```
uv run python main.py climatology --camels_root data/CAMELS --models_box path/to/runs --basins_file basins_test --hp 1 --ref_period_clim 19901001 19901030 --nproc_bv 3
```

- Hindcast
```
uv run python main.py hindcast --camels_root data/CAMELS --models_box path/to/runs --basins_file basins_test --hp 1 --ref_period_clim 19901001 19901030 --nproc_bv 3
```

### 2.2. Launch from a cmd line IDE such as pycharm

- Climatology
```
python main.py climatology --camels_root data/CAMELS --models_box path/to/runs --basins_file basins_test --hp 1 --ref_period_clim 19901001 19901030 --nproc_bv 3
```

- Hindcast
```
python main.py hindcast --camels_root data/CAMELS --models_box path/to/runs --basins_file basins_test --hp 1 --ref_period_clim 19901001 19901030 --nproc_bv 3
```

## 3. Output files
The output files are saved like
 - `path/to/runs/run_xxxx/*seedSSS_clim_hp[1-7].p` for climatology runs, see the **clim** string
 - `path/to/runs/run_xxxx/*seedSSS_hcst_hp[1-7].p` for hindcast runs, see the **hcst** string

From that point, only postprocessing remains.

## 4. Post-process runs

### 4.1. Runs from provided
For simplicity, you can use directly the post-processed files found in the fub folder `data_paper/processed/us/`
If you need to re-run from scratch, you will need the section 4.2. below

## 4.2. Post-process your runs 

These outputs need to be reformatted from *seedSSS_clim_hpX.p to:
 - 1. A Date indexed one-file-per-basin-per-hp with mean on seed. Save in `data/hindcast_bm/lstm/hp[1-7]/basin.csv`, 
   or `data/climato_bm/lstm/hp[1-7]/basin.csv`.
 - 2. A multi index dataframe with One-file-per-hp for all basin, where multi-index should be `(context, basin, hp, year, seed, Date)` and the column will be `(prediction)`. 
    And saved it in a FILE.parquet.gzip for faster processing, with:
   - context="lstm"
   - basin: 8-digit string ID of basin
   - hp: integer of lead time
   - seed: integer of the number of the seeds [1 to N]
   - year: Number of the member [1 to M], and -1 in the case of the deterministic (perfect) case
   - Date: yyyy-mm-dd date format
   - FILE: 
     - climatology : `lstm_hp[1-7]_CLIM56.parquet.gzip`
     - hindcast: `lstm_hp[1-7]_HIND56.parquet.gzip`
     - perfect or deterministic: `lstm_hp[1-7]_PERF531.parquet.gzip`
   - save like in : `~/data_paper/processed/us/FILE.parquet.gzip` 


[Note] The MLP runs use the `hindcast_bm/lstm/hp[1-7]/basin.csv` to perform the DA2 and the DA3 strategies

## 6. Notes
Note that the regional lstm runs are dropped in the  `data_paper/runs/lstm_mse_with_static_us` sub folder. They concern only LSTM 
trained with static inputs and the MSE loss function. These experiments can also be implemented for all the remain runs.

## End of our changes

---

## START of the original README file


# Catchment-Aware LSTMs for Regional Rainfall-Runoff Modeling

Accompanying code for our HESS paper "Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"

```
Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: Towards learning 
universal, regional, and local hydrological behaviors via machine learning applied to large-sample 
datasets, Hydrol. Earth Syst. Sci., 23, 5089–5110, https://doi.org/10.5194/hess-23-5089-2019, 2019. 
```

The manuscript can be found here (publicly available): [Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets](https://www.hydrol-earth-syst-sci.net/23/5089/2019/hess-23-5089-2019.html)

The code in this repository was used to produce all results and figures in our manuscript.


## Content of the repository

- `main.py` Main python file used for training and evaluating of our models, as well as to perform the robustness analysis
- `data/` contains the list of basins (USGS gauge ids) considered in our study
- `papercode/` contains the entire code (beside the in the root directory `main.py` file)
- `notebooks/` contain three notebooks, guiding through the results of our study. These notebooks should probably be your starting point.
    - `notebooks/performance.ipynb`: In this notebook, our modeling results are evaluated and compared against the benchmark models. All numbers and figures of the first two subsections of the results can be found here.
    - `notebooks/ranking.ipynb`: In this notebook, you can find the derivation of the feature ranking and the model robustness plot of the third subsection of the results.
    - `notebooks/embedding.ipynb`: In this notebook, you can find the analysis of the catchment embedding learned by our model as well as the cluster analysis. Here you find everything of the last subsection of the results.

## Setup to run the code locally

Download this repository either as zip-file or clone it to your local file system by running

```
git clone git@github.com:kratzert/ealstm_regional_modeling.git
```

### Setup Python environment
Within this repository we provide two environment files (`environment_cpu.yml` and `environment_gpu.yml`) that can be used with Anaconda or Miniconda to create an environment with all packages needed.

Simply run

```
conda env create -f environment_cpu.yml
```
for the cpu-only version. Or run

```
conda env create -f environment_gpu.yml
```
if you have a CUDA capable NVIDIA GPU. This is recommended if you want to train/evaluate the models on you machine but not strictly necessary. 

However, it is not strictly needed to re-train or re-evaluate any of the models to run the notebooks. Just make sure to download our pre-trained models and the pre-calculated model evaluations.

## Data needed

### Required Downloads

First of all you need the CAMELS data set, to run any of your code. This data set can be downloaded for free here:

- [CAMELS: Catchment Attributes and Meteorology for Large-sample Studies - Dataset Downloads](https://ral.ucar.edu/solutions/products/camels) Make sure to download the `CAMELS time series meteorology, observed flow, meta data (.zip)` file, as well as the `CAMELS Attributes (.zip)`. Extract the data set on your file system and make sure to put the attribute folder (`camels_attributes_v2.0`) inside the CAMELS main directory.

However, we trained our models with an updated version of the Maurer forcing data, that is still not published officially (CAMELS data set will be updated soon). The updated Maurer forcing contain daily minimum and maximum temperature. The original Maurer data included in the CAMELS data set only includes daily mean temperature. You can find the updated forcings temporarily here:

- [Updated Maurer forcing with daily minimum and maximum temperature](https://www.hydroshare.org/resource/17c896843cf940339c3c3496d0c1c077/)

Download and extract the updated forcing into the `basin_mean_forcing` folder of the CAMELS data set and do not rename it (name should be `maurer_extended`).

Next you need the simulations of all benchmark models. These can be downloaded from HydroShare under the following link:

- [CAMELS benchmark models](http://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1)

### Optional Downloads

To use our pre-trained models for evaluation or your own experiments, download the model files here:

- [Pre-trained models](http://www.hydroshare.org/resource/83ea5312635e44dc824eeb99eda12f06)

This download also contains the pre-evaluated model simulations of all our models.

## Run locally

For training or evaluating any of the models a CUDA capable NVIDIA GPU is recommended but not strictly necessary. Since we only train/use LSTM-based models a strong, multi-core CPU will work as well.

Before starting to do anything, make sure you have activated the conda environment.

```
conda activate ealstm
```

### Train model
To train a model, run the following line of code from the terminal

```
python main.py train --camels_root /path/to/CAMELS
```
This would train a single EA-LSTM model with a randomly generated seed using the basin average NSE as loss function and store the results under `runs/`. Additionally the following options can be passed:

- `--seed NUMBER` Train a model using a fixed random seed
- `--cache_data True` Load the entire training data into memory. This will speed up training but requires approximately 50GB of RAM.
- `--num_workers NUMBER` Defines the number of parallel threads that will load and preprocess inputs.
- `--no_static True` If passed, will train a standard LSTM without static features. If this is not desired, don't pass `False` but instead remove the argument entirely.
- `--concat_static True` If passed, will train a standard LSTM where the catchment attributes as concatenated at each time step to the meteorological inputs. If this is not desired, don't pass `False` but instead remove the argument entirely.
- `--use_mse True` If passed, will train the model using the mean squared error as loss function. If this is not desired, don't pass `False` but instead remove the argument entirely.

### Evaluate model

To evaluate a model, once training is finished, run the following line of code from the terminal.

```
python main.py evaluate --camels_root /path/to/CAMELS --run_dir path/to/model_run
```
This will calculate the discharge simulation for the validation period and store the results alongside the observed discharge for all basins in a pickle file. The pickle file is stored in the main directory of the model run.

### Evaluate robustness

To evaluate the model robustness against noise of the static input features run the following line of code from the terminal.

```
python main.py eval_robustness --camels_root /path/to/CAMELS --run_dir path/to/model_run
```

This will run 265,500 model evaluations (10 levels of added random noise and 50 repetitions per noise level for 531 basins). This evaluations is only implemented for our EA-LSTM. Therefore, make sure that the `model_run` folder contains the results of training an EA-LSTM.

### Run notebooks

In your terminal, go to the project folder and start a jupyter notebook server by running

```
jupyter notebook
```


## Citation

If you use any of this code in your experiments, please make sure to cite the following publication

```
@article{kratzert2019universal,
author = {Kratzert, F. and Klotz, D. and Shalev, G. and Klambauer, G. and Hochreiter, S. and Nearing, G.},
title = {Towards learning universal, regional, and local hydrological behaviors via machine learning 
applied to large-sample datasets},
journal = {Hydrology and Earth System Sciences},
volume = {23},
year = {2019},
number = {12},
pages = {5089--5110},
url = {https://www.hydrol-earth-syst-sci.net/23/5089/2019/},
doi = {10.5194/hess-23-5089-2019}
}
```

## License of our code
[Apache License 2.0](https://github.com/kratzert/ealstm_regional_modeling/blob/master/LICENSE)

## License of the updated Maurer forcings and our pre-trained models
The CAMELS data set only allows non-commercial use. Thus, our pre-trained models and the updated Maurer forcings underlie the same [TERMS OF USE](https://www2.ucar.edu/terms-of-use) as the CAMELS data set. 
