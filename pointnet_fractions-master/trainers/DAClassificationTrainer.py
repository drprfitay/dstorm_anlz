import os
import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from datasets.DstormDataset import DstormDataset

Blue = lambda x: '\033[94m' + x + '\033[0m'

class DAClassificationTrainer:
    # _persist_methods = ['print_to_log', 'json_parser', 'config_train_dir', 'start_timer',
    #                     'get_time_elapsed', 'get_current_datetime']

    def __init__(self, setup):

        self._setup = setup
        self.print_to_log = self._setup.print_to_log
        self.json_parser = self._setup.json_parser
        self.config_train_dir = self._setup.config_train_dir
        self.start_timer = self._setup.start_timer
        self.get_time_elapsed = self._setup.get_time_elapsed
        self.get_current_datetime = self._setup.get_current_datetime

        # Configuration params
        self.name = None
        self.set_multiple_label_classifier = False
        self.batch_size = 32
        self.num_epochs = 20
        self.lr = 1e-4
        self.skip_scheduler = True
        self.sch_milestones = [30]
        self.sch_gamma = 0.1
        self.workers = 4
        self.data_augmentation = False

        self.datasets = None
        self.dataloaders = None

        self._optimizer = optim.Adam
        self.optimizer = None

        self._scheduler = optim.lr_scheduler.MultiStepLR
        self.scheduler = None

        self.output_dir = self._setup.output_dir

    # def __getattr__(self, attribute):
    #     if attribute in self._persist_methods:
    #         return getattr(self._setup, attribute)
    #     raise AttributeError("{} doesn't have attribute {}".format(self.__module__, attribute))

    def __call__(self, config, classifier, dataset_list):
        self.print_to_log()
        self.print_to_log("Domain Adaptation Classification Trainer has been called")
        self._make_datasets(dataset_list)
        results = []
        best_classifiers = []
        for name in self._parse_config(config):
            classifier_to_train = copy.deepcopy(classifier)
            self._set_optimizer_and_scheduler(classifier_to_train)
            r, b = self._train(classifier_to_train)
            results.append(r)
            best_classifiers.append(b)
        return results, best_classifiers

    def _parse_config(self, config=None):
        config_path = os.path.abspath(os.path.join(self.config_train_dir, config))
        for config_name in self.json_parser(self, config_path):
            self.name = config_name
            yield config_name

    def _make_datasets(self, dataset_list):
        self.print_to_log("Creating Train and Validation datasets and loaders from {}".format(dataset_list))
        self.datasets = []
        self.dataloaders = []
        for dataset in dataset_list:
            self.datasets.append({x: DstormDataset(self._setup, name=dataset, split=x) for x in ['train', 'valid']})

            self.dataloaders.append({x: DataLoader(self.datasets[-1][x], batch_size=self.batch_size, shuffle=True,
                                                   num_workers=int(self.workers))
                                     for x in ['train', 'valid']})

    def _set_optimizer_and_scheduler(self, classifier):
        self.optimizer = self._optimizer(classifier.parameters(), lr=self.lr)
        self.scheduler = self._scheduler(self.optimizer, self.sch_milestones, self.sch_gamma)

    def _train(self, classifier):
        self.print_to_log('Training Classifier: {}. Train Datasets: {}.'
                          .format(classifier.name, ', '.join([x['train'].name for x in self.datasets])))

        # Saving classifier training params
        classifier.npoint_input = self.datasets[0]['train'][0][0].size(0)

        classifier.classifier_output_head = None

        total_training = []
        total_validation = []
        classifier.train_dataset = []
        classifier.classes_dict = []
        num_classes_list = []
        for dataset in self.datasets:
            if classifier.feature_extractor.pf_dim != dataset['train'][0][0].size(1) - 3:
                raise ValueError("Dimensions of point features input to the classifier {} doesn't match dataset {}"
                                 .format(classifier.pf_dim, dataset['train'].name))

            if classifier.npoint_input != dataset['train'][0][0].size(0):
                self.print_to_log("Different number of points input between datasets. Expected {} got dataset {}:{}"
                                  .format(classifier.npoint_input, dataset['train'].name, dataset['train'][0][0].size(0)))

            if not self.set_multiple_label_classifier:
                if classifier.num_classes != dataset['train'].num_classes:
                    raise ValueError("Number of classes output by the classifier {} doesn't match dataset {}"
                                     .format(classifier.num_classes, dataset['train'].name))

            total_training.append(len(dataset['train']))
            total_validation.append(len(dataset['valid']))

            # Saving classifier training params
            classifier.train_dataset.append(dataset['train'].name)
            classifier.classes_dict.append(dataset['train'].classes)
            num_classes_list.append(dataset['train'].num_classes)

        if self.set_multiple_label_classifier:
            classifier.set_multiple_label_classifiers(num_classes_list)
            self._set_optimizer_and_scheduler(classifier)

        self.print_to_log('number of samples to train (all domains): {} = {}'
                          .format("+".join(map(lambda x: " {} ".format(x), total_training)), sum(total_training)))
        self.print_to_log('number of samples to validate (all domains): {} = {}'
                          .format("+".join(map(lambda x: " {} ".format(x), total_validation)), sum(total_validation)))

        self.print_to_log('optimizer {} with learning rate {}'.format(self.optimizer.__module__, self.lr))
        if not self.skip_scheduler:
            self.print_to_log('scheduler at epochs {}, gamma={}'.format(self.sch_milestones, self.sch_gamma))

        classifier.cuda()

        best_classifier = classifier.export_network()
        best_epoch = 0
        best_accuracy = 0.0

        v_loss = {x: [] for x in ['train', 'valid']}

        self.print_to_log("Start training...")
        self.start_timer()
        for epoch in range(1, self.num_epochs + 1):
            print("--------Epoch {}--------".format(epoch))

            # Each epoch has training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    classifier.train()
                    torch.set_grad_enabled(True)
                else:
                    classifier.eval()
                    torch.set_grad_enabled(False)

                running_domain_loss = 0
                running_label_loss = 0
                running_accuracy = 0

                # Iterate over data
                num_batches = min([len(dataloader[phase]) for dataloader in self.dataloaders])
                dataloaders_list = (dataloader[phase] for dataloader in self.dataloaders)
                batches = zip(*dataloaders_list)

                for batch_idx, data in enumerate(batches, 0):
                    pointcloud, radius, labels_list = [[], [], []]
                    domain = torch.empty(0, dtype=torch.long)
                    splits = [0]
                    for domain_num, domain_data in enumerate(data, 0):
                        pointcloud.append(domain_data[0].permute(0, 2, 1))
                        radius.append(domain_data[1].unsqueeze(1))
                        labels_list.append(domain_data[2])
                        domain = torch.cat([domain, torch.zeros(len(labels_list[-1]), dtype=torch.long) + domain_num])
                        splits.append(splits[-1] + len(labels_list[-1]))

                    pointcloud = torch.cat(pointcloud)
                    radius = torch.cat(radius)
                    labels = torch.cat(labels_list)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward - track history only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        pointcloud, radius, domain = pointcloud.cuda(), radius.cuda(), domain.cuda()
                        label_preds, domain_preds = classifier(pointcloud, radius)

                        domain_loss = classifier.domain_criterion(domain_preds, domain)
                        running_domain_loss += domain_loss  # Statistics

                        if classifier.single_label_classifier:
                            labels = labels.cuda()
                            label_loss = classifier.label_criterion(label_preds, labels)
                            loss = domain_loss + label_loss

                            # Statistics
                            running_label_loss += label_loss  # Statistics
                            label_decisions = label_preds.max(1)[1]
                            running_accuracy += label_decisions.eq(labels).float().mean()

                        else:
                            label_loss_vector = torch.empty(0, dtype=torch.float)
                            label_accuarcy_vector = torch.empty(0, dtype=torch.float)
                            loss = domain_loss
                            for ind, labels in enumerate(labels_list, 0):
                                labels = labels.cuda()
                                preds = label_preds[ind][splits[ind]:splits[ind+1]]
                                label_loss = classifier.label_criterion(preds, labels)
                                label_loss_vector = torch.cat([label_loss_vector, label_loss.cpu().unsqueeze(0)])
                                loss += label_loss

                                # Statistics
                                label_decisions = preds.max(1)[1]
                                label_accuracy = label_decisions.eq(labels).float().mean().cpu().unsqueeze(0)
                                label_accuarcy_vector = torch.cat([label_accuarcy_vector, label_accuracy])

                            running_label_loss += label_loss_vector
                            running_accuracy += label_accuarcy_vector

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                epoch_domain_loss = running_domain_loss / num_batches
                epoch_label_loss = running_label_loss / num_batches
                epoch_accuracy = running_accuracy / num_batches

                v_dict = {
                    "epoch_domain_loss" : epoch_domain_loss.cpu(),
                    "epoch_label_loss"  : epoch_label_loss.cpu(),
                    "epoch_accuracy"    : epoch_accuracy.cpu()
                }
                v_loss[phase].append(v_dict)

                mean_epoch_accuracy = epoch_accuracy.mean()
                mean_label_loss = epoch_label_loss.mean()
                if phase == 'valid' and mean_epoch_accuracy > best_accuracy:
                    best_epoch = epoch
                    best_accuracy = mean_epoch_accuracy
                    best_classifier = classifier.export_network()
                    print(Blue("{} Domain Loss: {:.4f} ; Label Loss: {:.4f} ; Accuracy: {:.2f}%"
                               .format(phase, epoch_domain_loss, mean_label_loss, mean_epoch_accuracy * 100.0)))
                else:
                    print("{} Domain Loss: {:.4f} ; Label Loss: {:.4f} ; Accuracy: {:.2f}%"
                          .format(phase, epoch_domain_loss, mean_label_loss, mean_epoch_accuracy * 100.0))

                if not classifier.single_label_classifier:
                    loss_list = ["{} : {:.4f}".format(i, z) for i, z in  enumerate(epoch_label_loss.tolist(),0)]
                    accuracy_list = ["{} : {:.4f}".format(i, z) for i, z in  enumerate(epoch_accuracy.tolist(),0)]
                    print("Epoch label loss:\t{}".format(loss_list))
                    print("Epoch label accuracy:\t{}".format(accuracy_list))

                if phase == 'train' and not self.skip_scheduler:
                    self.scheduler.step()

        self.print_to_log('\nFinish training in %s' % (self.get_time_elapsed()))
        best_acc = int(best_accuracy * 100.0)
        self.print_to_log('Best Validation accuracy: {:.2f}% Epoch: {}\n'.format(best_acc, best_epoch))

        results = {}
        results["name"] = '{}_best_acc_{}_{}'.format(self.name, round(best_acc), self.get_current_datetime())
        results["best_classifier"] = best_classifier
        results["v_loss"] = v_loss

        self.print_to_log("Saving results: {}".format(results["name"]))
        torch.save(results, os.path.join(self.output_dir, results["name"]))

        classifier = classifier.cpu()
        classifier.load_network(best_classifier)

        self.print_to_log()
        return results, classifier
