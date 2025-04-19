'''
We implemented `iCaRL+RMM`, `FOSTER+RMM` in [rmm.py](models/rmm.py).  We implemented the `Pretraining Stage` of `RMM` in [rmm_train.py](rmm_train.py). 
Use the following training script to run it.
```bash
python -m pycil2.rmm_train --config=./exps/rmm-pretrain.json
```
'''
import json
import argparse
from pycil2.trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of RMM-FOSTER and RMM-iCaRL.')
    parser.add_argument('--config', type=str, default='./exps/rmm-pretrain.json',
                        help='Json file of settings.')

    return parser


if __name__ == '__main__':
    main()