# Temporal Fusion Transformers for Interpretable Prediction of Mutual Fund Return

Authors: Xuxi (Norman) Guo

## Code Organisation
This repository contains the source code for the Temporal Fusion Transformer, along with the training and evaluation routines for the experiments described in the paper.

The key modules for experiments are organised as:

* **data\_formatters**: Stores the main dataset-specific column definitions, along with functions for data transformation and normalization. For compatibility with the TFT, new experiments should implement a unique ``GenericDataFormatter`` (see **base.py**), with examples for the default experiments shown in the other python files.
* **expt\_settings**: Holds the folder paths and configurations for the default experiments,
* **libs**: Contains the main libraries, including classes to manage hyperparameter optimisation (**hyperparam\_opt.py**), the main TFT network class (**tft\_model.py**), and general helper functions (**utils.py**)

Scripts are all saved in the main folder, with descriptions below:

* **run.sh**: Simple shell script to ensure correct environmental setup.
* **script\_download\_data.py**: Downloads data for the main experiment and processes them into csv files ready for training/evaluation.
* **script\_train\_fixed\_params.py**: Calibrates the TFT using a predefined set of hyperparameters, and evaluates for a given experiment.
* **script\_hyperparameter\_optimisation.py**: Runs full hyperparameter optimization using the default random search ranges defined for the TFT.

## Running Default Experiements
Our four default experiments are divided into ``volatility``, ``electricity``, ``traffic``, and``favorita``. To run these experiments, first download the data, and then run the relevant training routine.

### Step 1: Download data for default experiments
To download the experiment data, run the following script:
```bash
python3 -m script_download_data $EXPT $OUTPUT_FOLDER
```
where ``$EXPT`` can be any of {``volatility``, ``electricity``, ``traffic``, ``favorita``}, and ``$OUTPUT_FOLDER`` denotes the root folder in which experiment outputs are saved.

### Step 2: Train and evaluate network
To train the network with the optimal default parameters, run:
```bash
python3 -m script_train_fixed_params $EXPT $OUTPUT_FOLDER $USE_GPU 
```
where ``$EXPT`` and ``$OUTPUT_FOLDER`` are as above, ``$GPU`` denotes whether to run with GPU support (options are {``'yes'`` or``'no'``}).

For full hyperparameter optimization, run:
```bash
python3 -m script_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes
```
where options are as above.

## Customising Scripts for New Datasets
To re-use the hyperparameter optimization scripts for new datasets, we need to add a new experiment -- which involves the creation of a new data formatter and config updates.

### Step 1: Implement custom data formatter
First, create a new python file in ``data_formatters`` (e.g. example.py) which contains a data formatter class (e.g. ``ExampleFormatter``). This should inherit ``base.GenericDataFormatter`` and provide implementations of all abstract functions. An implementation example can be found in volatility.py.

### Step 2: Update configs.py
Add a name for your new experiement to the ``default_experiments`` attribute in ``expt_settings.configs.ExperimentConfig`` (e.g. ``example``).
```python
default_experiments = ['volatility', 'electricity', 'traffic', 'favorita', 'example']
```


Next, add an entry in ``data_csv_path`` mapping the experiment name to name of the csv file containing the data:

```python
@property
  def data_csv_path(self):
    csv_map = {
        'volatility': 'formatted_omi_vol.csv',
        'electricity': 'hourly_electricity.csv',
        'traffic': 'hourly_data.csv',
        'favorita': 'favorita_consolidated.csv',
        'example': 'mydata.csv'  # new entry here!
    }

    return os.path.join(self.data_folder, csv_map[self.experiment])
```

Lastly, add your custom data formatter to the factory function:

```python
def make_data_formatter(self):
    """Gets a data formatter object for experiment.

    Returns:
      Default DataFormatter per experiment.
    """

    data_formatter_class = {
        'volatility': data_formatters.volatility.VolatilityFormatter,
        'electricity': data_formatters.electricity.ElectricityFormatter,
        'traffic': data_formatters.traffic.TrafficFormatter,
        'example': data_formatters.example.ExampleFormatter, # new entry here!
    }
```

As an optional step, change the number of random search iterations if required:
```python
@property
  def hyperparam_iterations(self):
    
    my_search_iterations=1000
    
    if self.experiment == 'example':
      return my_search_iterations
    else:
      return 240 if self.experiment == 'volatility' else 60
```


### Step 3: Run training script
Full hyperparameter optimization can then be run as per the previous section, e.g.:
```bash
python3 -m script_hyperparam_opt example . yes yes

```
