# CUDA_VISIBLE_DEVICES=0 python setup.py
import os
import importlib
import pkgutil
from datetime import datetime
import torch

import random
import numpy as np

import utils.MiscUtils
import utils.torch_pointcloud_utils

import data_tools.DatasetBuilder
import data_tools.DatasetSpliter

import datasets.DstormDataset

import models.PointNet
import models.Classifiers
import models.DomainAdaptation

import trainers.ClassificationTrainer
import trainers.DAClassificationTrainer

import testers.ClassificationTester

class Setup:
    def __init__(self, output_dir_name=None):

        self.base_path = '../../'

        # Data dirs
        self.root_train_dir = os.path.join(self.base_path, 'data/train_data')
        self.root_test_dir = os.path.join(self.base_path, 'data/test_data')
        self.root_training_datasets_dir = os.path.join(self.base_path, 'data/training_datasets')

        # Config dirs
        self.config_build_dir = os.path.join(self.base_path, 'config/build')
        self.config_train_dir = os.path.join(self.base_path, 'config/train')
        self.config_test_dir = os.path.join(self.base_path,'config/test')

        # Output dir
        self.root_output_path = os.path.join(self.base_path, 'output')
        self.output_dir = None
        self.create_output_dir(output_dir_name)

        # Open log
        self.log = None
        self.open_log()

        # Utils
        self.plotter = utils.MiscUtils.Plotter
        self.plot_loss_functions = self.plotter.plot_loss_functions

        # Setting random seeds
        self.manualSeed = random.randint(1, 10000)  # fix seed
        self.print_to_log("Random Seed: {}".format(self.manualSeed))
        random.seed(self.manualSeed)
        np.random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)

        self.random_shuffle = random.shuffle
        self.np_random_normal = np.random.normal
        self.np_random_uniform = np.random.uniform
        self.torch_random_perm = torch.randperm

        self.reload()

    def __getattr__(self, attribute):
        raise AttributeError("{} doesn't have attribute {}".format(self.__module__, attribute))

    def __del__(self):
        self.print_to_log("Deleting setup object")

    def clear_cuda(self):
        torch.cuda.empty_cache()

    def create_output_dir(self, output_dir_name=None):

        if output_dir_name is None:
            self.output_dir = os.path.join(self.root_output_path, datetime.now().strftime("%d_%m_%Y"))
        else:
            self.output_dir = os.path.join(self.root_output_path, output_dir_name)

        # If output_dir already exist create another one
        if os.path.exists(self.output_dir):
            i = 1
            while os.path.exists(self.output_dir + '_{}'.format(i)):
                i += 1
            self.output_dir = self.output_dir + '_{}'.format(i)

        os.mkdir(self.output_dir)

    def open_log(self):
        self.log = utils.MiscUtils.Log(self)
        self.log.print_to_log("Open log at {}".format(os.path.abspath(self.output_dir)))

        self.__setattr__('print_to_log', self.log.print_to_log)
        self.__setattr__('get_current_datetime', self.log.get_current_datetime)
        self.__setattr__('start_timer', self.log.start_timer)
        self.__setattr__('get_time_elapsed', self.log.get_time_elapsed)

    def reload(self):

        # utils
        importlib.reload(utils.MiscUtils)
        importlib.reload(utils.torch_pointcloud_utils)
        self.utils()

        # data tools
        importlib.reload(data_tools.DatasetBuilder)
        importlib.reload(data_tools.DatasetSpliter)
        self.data_tools()

        # datasets
        importlib.reload(datasets.DstormDataset)
        self.datasets()

        # models
        importlib.reload(models.PointNet)
        importlib.reload(models.Classifiers)
        importlib.reload(models.DomainAdaptation)
        self.models()

        # trainers
        importlib.reload(trainers.ClassificationTrainer)
        importlib.reload(trainers.DAClassificationTrainer)
        self.trainers()

        # testers
        importlib.reload(testers.ClassificationTester)
        self.testers()

    def utils(self):
        self.__setattr__('parser', utils.MiscUtils.Parser(self))
        self.__setattr__('json_parser', self.parser.json_parser)
        self.__setattr__('pointcloud_utils', utils.torch_pointcloud_utils.PointCloudUtils(self))

    def data_tools(self):
        self.__setattr__('dataset_builder', data_tools.DatasetBuilder.DatasetBuilder(self))
        self.__setattr__('dataset_spliter', data_tools.DatasetSpliter.DatasetSpliter(self))

    def datasets(self):
        self.__setattr__('DstormDataset', datasets.DstormDataset.DstormDataset)

    def models(self):
        self.__setattr__('pointnet_classifier', models.Classifiers.PointNetClassifier(self))
        self.__setattr__('da_pointnet_classifier', models.DomainAdaptation.DAClassifier(self))
        # self.da_pointnet_classifier = models.DomainAdaptation.DAClassifier(self)

    def trainers(self):
        self.__setattr__('classification_trainer', trainers.ClassificationTrainer.ClassificationTrainer(self))
        self.__setattr__('da_classification_trainer', trainers.DAClassificationTrainer.DAClassificationTrainer(self))

    def testers(self):
        self.__setattr__('classification_tester', testers.ClassificationTester.ClassificationTester(self))

    def train_classic(self):
        config = 'edens_a53t_512_classifier_train_config.json'
        self.edens_a53t_train_dataset = self.DstormDataset(self, 'edens_a53t_512', split='train')
        self.edens_a53t_valid_dataset = self.DstormDataset(self, 'edens_a53t_512', split='valid')
        self.edens_a53t_classifier_train_results, self.edens_a53t_classifier =\
            self.classification_trainer(
                config,
                self.pointnet_classifier,
                self.edens_a53t_train_dataset,
                self.edens_a53t_valid_dataset
            )

    def train_da(self):
        config = 'sa_edens_a53t_tdp_512_classifier_train_config.json'


s = Setup()
from IPython import embed
embed()
