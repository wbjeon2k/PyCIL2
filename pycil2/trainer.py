import sys
import logging
import copy
import torch
from pycil2.utils import factory
from pycil2.utils.data_manager import DataManager
from pycil2.utils.toolkit import count_parameters
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["device"]
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {}, {}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}, NME: {}".format(cnn_accy, nme_accy))

            cnn_curve[task] = cnn_accy
            nme_curve[task] = nme_accy
            logging.info("CNN curve: {}".format(cnn_curve))
            logging.info("NME curve: {}".format(nme_curve))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy))

            cnn_curve[task] = cnn_accy
            logging.info("CNN curve: {}".format(cnn_curve))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(device))
        gpus.append(device)

    args["device"] = gpus[0]
    if len(gpus) > 1:
        args["gpus"] = gpus
    else:
        args["gpus"] = [args["device"]]


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
