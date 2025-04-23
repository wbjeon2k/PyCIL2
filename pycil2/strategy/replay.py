import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pycil2.strategy.base import BaseLearner
from pycil2.utils.toolkit import tensor2numpy
from pycil2.utils.data_manager import DataManager
EPSILON = 1e-8


class Replay(BaseLearner):
    """Replay strategy for class-incremental learning.
    
    This implementation allows working with any nn.Module rather than
    requiring a specific network type. A model must be set using
    set_model() before training.
    """
    
    def __init__(self, args):
        super().__init__(args)

    def after_task(self):
        """Update known classes and log exemplar info after a task."""
        self._prev_known_classes = self._known_classes
        self._prev_total_classes = self._total_classes
        
        
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def update_classifier(self, nb_classes):
        """Update the classifier to handle new classes.
        
        This method should be implemented according to the model type.
        It might add new output neurons or adapt existing ones.
        
        Args:
            nb_classes: Total number of classes after update
        """
        # This is a placeholder that needs to be implemented
        # based on the specific model being used
        # raise NotImplementedError(
        #     "Please implement update_classifier based on your model type"
        # )
        pass

    def incremental_train(self, data_manager : DataManager):
        """Train the model incrementally with replay memory.
        
        Args:
            data_manager: Data manager providing access to task data
        """
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        
        # Update classifier to handle the new total number of classes
        self.update_classifier(self._total_classes)
        
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # Prepare data loaders with memory replay
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"]
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"]
        )

        # Handle multi-GPU training
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            
        # Train the model
        self._train(self.train_loader, self.test_loader)

        # Update memory with exemplars
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        
        # Restore from DataParallel if needed
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        """Train the model on current task data.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test/validation data
        """
        self._network.to(self._device)
        
        if self._cur_task == 0:
            # Initial task training
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            # Subsequent tasks with replay
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """Initial training procedure for the first task.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test/validation data
            optimizer: Optimizer to use for training
            scheduler: Learning rate scheduler
        """
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                
                # Extract logits if output is a dict
                if isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Standard cross-entropy loss
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # Calculate accuracy
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # Log progress
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """Training procedure for subsequent tasks with replay.
        
        Args:
            train_loader: DataLoader with current task data + replay memory
            test_loader: DataLoader for test/validation data
            optimizer: Optimizer to use for training
            scheduler: Learning rate scheduler
        """
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                
                # Extract logits if output is a dict
                if isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Standard cross-entropy loss (no separation between old and new classes)
                loss_clf = F.cross_entropy(logits, targets)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # Calculate accuracy
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            # Log progress
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)