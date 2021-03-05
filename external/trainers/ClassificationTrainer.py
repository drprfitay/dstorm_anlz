import os
import numpy as np
import copy
import torch
import torch.optim as optim

from utils.MiscUtils import Parser, Log
from datasets.DstormDataset import DstormDataset

Blue = lambda x: '\033[94m' + x + '\033[0m'


class ClassificationTrainer:

    def __init__(self, log_path=None):
        self.parser = Parser()
        self.log = Log(log_path)

    class Config:
        def __init__(self):
            self.name = None
            self.num_epochs = 20

            # Optimizer
            self._optimizer = optim.Adam
            self.lr = 1e-4
            self.optimizer = None

            # Scheduler
            self._scheduler = optim.lr_scheduler.MultiStepLR
            self.skip_scheduler = True
            self.sch_milestones = [30]
            self.sch_gamma = 0.1
            self.scheduler = None

    def run_from_config(self, config_path, classifier, trainloader, validloader=None, device=None, output_dir=None):
        print(f"Classification Trainer has been called")
        results = {}
        config = self.Config()
        for name in self.parser.json_parser(config, config_path):
            print(f"Train {name}")
            classifier_to_train = copy.deepcopy(classifier)
            self._set_optimizer_and_scheduler(config, classifier_to_train)
            results[name], _ = self.train(
                classifier=classifier,
                trainloader=trainloader,
                optimizer=config.optimizer,
                num_epochs=config.num_epochs,
                device=device,
                validloader=validloader,
                scheduler=config.scheduler,
                output_dir=output_dir
            )

        return results

    @staticmethod
    def _set_optimizer_and_scheduler(config, classifier):
        config.optimizer = config._optimizer(classifier.parameters(), lr=config.lr)
        if not config.skip_scheduler:
            config.scheduler = config._scheduler(config.optimizer, config.sch_milestones, config.sch_gamma)

    def train(self, classifier, trainloader, optimizer, num_epochs,
              device=None, validloader=None, scheduler=None, output_dir=None):

        now = self.log.get_current_datetime()
        dataset_name = os.path.basename(trainloader.dataset.root)
        lr = optimizer.param_groups[-1]["lr"]

        print(f'{now} Training Classifier: {classifier.name}. Train Dataset: {dataset_name}.')
        print(f'optimizer {optimizer.__module__} with learning rate {lr}')

        if scheduler is not None:
            print('Scheduler is {}'.format(scheduler.__module__))

        if classifier.num_classes != trainloader.sampler.num_classes:
            raise ValueError("Number of classes output by the classifier {} doesn't match trainloader"
                             .format(classifier.num_classes))

        if classifier.feature_extractor.pf_dim != trainloader.dataset[0][0].size(1) - 3:
            raise ValueError("Dimensions of point features input to the classifier {} doesn't match training dataset"
                             .format(classifier.pf_dim))

        # Saving classifier training params
        classifier.train_dataset = trainloader.dataset.root
        classifier.classes_dict = trainloader.sampler.classes
        classifier.npoint_input = trainloader.dataset[0][0].size(0)

        device = device if device is not None else torch.device("cpu")
        classifier.to(device)

        best_classifier = classifier.export_network()
        best_epoch = 0
        best_acc = 0.0
        best_loss = 100.0

        if validloader is not None:
            loaders = {
                'train': trainloader,
                'valid': validloader
            }
        else:
            loaders = {
                'train': trainloader
            }

        v_loss = {x: [] for x in loaders.keys()}

        print("Start training...")
        self.log.start_timer()
        for epoch in range(1, num_epochs + 1):
            print("--------Epoch {}--------".format(epoch))

            # Each epoch has training and validation phase
            for phase in loaders:
                if phase == 'train':
                    classifier.train()
                    torch.set_grad_enabled(True)
                else:
                    classifier.eval()
                    torch.set_grad_enabled(False)

                running_loss = 0
                running_correct = 0

                # Iterate over data
                for batch_idx, data in enumerate(loaders[phase], 0):
                    pointcloud, radius, label = data
                    pointcloud = pointcloud.permute(0, 2, 1)
                    radius = radius.unsqueeze(1)
                    pointcloud, radius, label = pointcloud.to(device), radius.to(device), label.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward - track history only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        output = classifier(pointcloud, radius)
                        pred = output.max(1)[1]
                        loss = classifier.criterion(output, label)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * pointcloud.size(0)
                    running_correct += pred.eq(label).sum().item()

                epoch_loss = running_loss / len(loaders[phase].sampler)
                epoch_acc = running_correct / len(loaders[phase].sampler)

                v_loss[phase].append(epoch_loss)

                if phase == 'valid' and epoch_acc > best_acc:
                    print(Blue("{} Loss: {:.4f}, Accuracy: {:.2f}%".format(phase, epoch_loss, epoch_acc * 100.0)))
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_classifier = classifier.export_network()
                else:
                    print("{} Loss: {:.4f}, Accuracy: {:.2f}%".format(phase, epoch_loss, epoch_acc * 100.0))

                if phase == 'train' and scheduler is not None:
                    self.scheduler.step()

        if validloader is None:
            print(Blue("Last epoch Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch_loss, epoch_acc * 100.0)))
            best_loss = epoch_loss
            best_acc = epoch_acc
            best_epoch = epoch
            best_classifier = classifier.export_network()

        print('\nFinish training in %s' % (self.log.get_time_elapsed()))
        best_acc = int(best_acc * 100.0)
        print('Best Validation accuracy: {:.2f}% Epoch: {}\n'.format(best_acc, best_epoch))

        results = {
            "classifier_name": f'{now}_{classifier.name}_{dataset_name}_best_acc_{round(best_acc)}',
            "classifier_dict": best_classifier,
            "train_time": now,
            "train_dataset": classifier.train_dataset,
            "epochs": num_epochs,
            "batch_size": trainloader.batch_size,
            "data_augmentation": trainloader.dataset.data_augmentation,
            "classes_dict": classifier.classes_dict,
            "npoint_train": classifier.npoint_input,
            "best_loss": best_loss,
            "best_accuracy": best_acc,
            "best_epoch": best_epoch,
            "v_loss": v_loss
        }

        if scheduler is not None:
            results = {**results, **scheduler.state_dict()}
        else:
            results["lr"] = lr

        if output_dir is not None:
            print("Saving results: {}".format(results["classifier_name"]))
            torch.save(results, os.path.join(output_dir, results["classifier_name"]))

        classifier = classifier.cpu()
        classifier.load_network(best_classifier)

        return results, classifier


if __name__ == '__main__':
    import random
    from torch.utils.data import DataLoader
    from datasets.samplers import MinSampler
    from models.Classifiers import PointNetClassifier

    manual_seed = random.randint(1, 10000)  # fix seed
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    device = torch.device("cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Seed:{manual_seed}')
    print(device)

    batch_size = 16
    dataset_path = '/home/oronlevy/ext/data/training_datasets/edens_a53t_2048'

    dataset = DstormDataset(dataset_path)
    sampler = MinSampler(dataset.df, classes_list=['Naive', 'Mutant'])
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                             sampler=sampler, pin_memory=False)

    classifier = PointNetClassifier(features_len=1024, num_classes=2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)

    trainer = ClassificationTrainer()
    r, c = trainer.train(
        classifier,
        trainloader,
        optimizer,
        device=device,
        num_epochs=30,
        validloader=None,
        scheduler=None,
        output_dir=None
    )
