# DANLI
Code for EMNLP 2022 Paper DANLI: Deliberative Agent for Following Natural Language Instructions [[paper](https://aclanthology.org/2022.emnlp-main.83/)] [[arXiv](https://arxiv.org/abs/2210.12485)]


## Installation
Create a virtual environment with Python 3.8 such as using `conda`:
```
conda create --name danli python=3.8
conda activate danli
```
Install the dependencies:
```
pip install -r requirements.txt
pip install -e .
```
Install the [fast-downward PDDL planner](https://www.fast-downward.org/HomePage):
```
cd ..
git clone https://github.com/aibasel/downward.git fast_downward
cd fast_downward && ./build.py
cd ../DANLI
```

## Download the TEACh dataset and the model weights
Download the raw dataset from the [official teach repo](https://github.com/alexa/teach) into `teach-dataset`. 

Download the pre-processed data:
```
sh download_data.sh
```
Download the model weights: 
```
sh download_model.sh
```

## Run DANLI
Set paths:
```
export DANLI_ROOT_DIR=$(pwd)
export DANLI_DATA_DIR=$DANLI_ROOT_DIR/teach-dataset
export DANLI_MODEL_DIR=$DANLI_ROOT_DIR/models
export DANLI_EVAL_DIR=$DANLI_ROOT_DIR/evals

# replace with your fast downward installation path
export FASTDOWNWARD_DIR=<YOUR_FAST_DOWNWARD_INSTALLATION_DIR>
```

Start an x-server (a prerequisite to launch the ai2thor environment) and set the DISPLAY variable:
```
sudo python3 start_x.py start 9
export DISPLAY=:9
```

Run the evaluation:
```
python run/run_neural_symbolic.py \
    --eval_name danli_eval
    --benchmark edh
    --split valid_seen
    --num_processes 2
    --num_gpus 2
```
Note that the above command runs DANLI for the `valid_seen` split on TEACh EDH benchmark by running 2 workers in parallel. The output will be stored under `$DANLI_EVAL_DIR/danli_eval`.

## Compute the metrics
```
teach_eval \
    --data_dir $DANLI_DATA_DIR \
    --inference_output_dir $DANLI_EVAL_DIR/danli_eval/predictions \
    --split divided_val_seen \
    --benchmark edh\
    --metrics_file $DANLI_EVAL_DIR/danli_eval/metrics/metrics
```
After running the evaluation for both the `valid_seen` and `valid_unseen` splits, the metrics on `divided_val_seen`, `divided_val_unseen`, `divided_test_seen` and `divided_test_unseen` can be computed through running the above command with the corresponding `split` argument. See [here](https://github.com/alexa/teach#teach-edh-offline-evaluation) for an explaination about the difference between the divided version of data split and the original ones. 

## Contact
Feel free to create an issue or send email to zhangyic@umich.edu