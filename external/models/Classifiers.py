import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.PointNet import PointNet2


class FCNClassifier(nn.Module):

    def __init__(self, in_channel, mlp):
        super(FCNClassifier, self).__init__()

        self.name = 'Classifier'
        self._in_channel = in_channel
        self._mlp = mlp

        self._batch_norm = True
        self._dropout = True

        self.fcn = nn.ModuleList()
        self.reset_network(False)

    def forward(self, x):
        for mlp in self.fcn:
            x = mlp(x)
        return x

    def export_network(self):
        # print("Exporting {}: {} -> {} . Batch Norm {} Dropout {}"
        #                   .format(self.in_channel, ' -> '.join(map(str, self.mlp)), self.batch_norm, self.dropout))
        classifier_dict = {}
        classifier_dict["name"] = self.name
        classifier_dict["in_channel"] = self.in_channel
        classifier_dict["mlp"] = self.mlp
        classifier_dict["batch_norm"] = self.batch_norm
        classifier_dict["dropout"] = self.dropout
        classifier_dict["state_dict"] = copy.deepcopy(self.state_dict())
        return classifier_dict

    def load_network(self, classifier_dict):
        self.name = classifier_dict["name"]
        self._in_channel = classifier_dict["in_channel"]
        self._mlp = classifier_dict["mlp"]
        self._batch_norm = classifier_dict["batch_norm"]
        self._dropout = classifier_dict["dropout"]
        self.reset_network(False)
        self.load_state_dict(classifier_dict["state_dict"])
        print("Loading {}: {} -> {} . Batch Norm {} Dropout {}"
                          .format(self.name, self.in_channel, ' -> '.join(map(str, self.mlp)), self.batch_norm, self.dropout))

    def reset_network(self, _print=True):
        if _print:
            print("Resetting Label Classifier: {} -> {} . Batch Norm {} Dropout {}"
                              .format(self.in_channel, ' -> '.join(map(str, self.mlp)), self.batch_norm, self.dropout))
        self.fcn = nn.ModuleList()
        last_channel = self._in_channel
        for ind, out_channel in enumerate(self._mlp, 1):
            self.fcn.append(nn.Linear(last_channel, out_channel))
            if ind != len(self._mlp):
                self.fcn.append(nn.ReLU())
                if self.batch_norm:
                    self.fcn.append(nn.BatchNorm1d(out_channel))
                if self.dropout:
                    self.fcn.append(nn.Dropout(0.4))
            last_channel = out_channel

    def set_output_channel(self, value):
        self._mlp[-1] = value
        self.reset_network()

    def set_batch_norm(self):
        self._batch_norm = True
        self.reset_network()

    def unset_batch_norm(self):
        self._batch_norm = False
        self.reset_network()

    def set_dropout(self):
        self._dropout = True
        self.reset_network()

    def unset_dropout(self):
        self._dropout = False
        self.reset_network()

    @property
    def in_channel(self):
        return self._in_channel

    @in_channel.setter
    def in_channel(self, value):
        print("Setting input channel to {}.".format(value))
        self._in_channel = value
        self.reset_network()

    @property
    def mlp(self):
        return self._mlp

    @mlp.setter
    def mlp(self, value):
        print("Setting input channel to {}.".format(value))
        self._mlp = value
        self.reset_network()

    @property
    def batch_norm(self):
        return self._batch_norm

    @property
    def dropout(self):
        return self._dropout


class PointNetClassifier(nn.Module):

    def __init__(self, features_len=1024, num_classes=3):
        super(PointNetClassifier, self).__init__()

        self._features_len = features_len
        self._num_classes = num_classes

        self._basename = "PointNetClassifier"
        self._name = "{}_Nclasses{}_FeaturesLen{}".format(self._basename, self.num_classes, self.features_len)

        # train params
        self.train_dataset = None
        self.classes_dict = None
        self.npoint_input = None
        self.criterion = nn.NLLLoss()

        self.feature_extractor = PointNet2(self.features_len)

        # 3 layers classification network: input features from pointcloud + radius ; output number of classes
        self.label_classifier = FCNClassifier(self.features_len + 1, [512, 256, self._num_classes])

    def load_feature_extractor(self, feature_extractor_dict):
        self.feature_extractor.load_network(feature_extractor_dict)

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def export_network(self):
        classifier_dict = {}
        classifier_dict["name"] = self.name
        classifier_dict["basename"] = self._basename
        classifier_dict["num_classes"] = self.num_classes
        classifier_dict["features_length"] = self.features_len
        classifier_dict["train_dataset"] = self.train_dataset
        classifier_dict["npoint_input"] = self.npoint_input
        classifier_dict["classes_dict"] = self.classes_dict
        classifier_dict["feature_extractor"] = self.feature_extractor.export_network()
        classifier_dict["label_classifier"] = self.label_classifier.export_network()
        # classifier_dict["state_dict"] = copy.deepcopy(self.state_dict())
        # print("Exporting PointNet Classifier {}".format(self.name))
        return classifier_dict

    def load_network(self, classifier_dict):
        print("Loading PointNet Classifier {}".format(self.name))
        self._name = classifier_dict["name"]
        self._basename = classifier_dict["basename"]
        self._num_classes = classifier_dict["num_classes"]
        self._features_len = classifier_dict["features_length"]
        self.train_dataset = classifier_dict["train_dataset"]
        self.npoint_input = classifier_dict["npoint_input"]
        self.classes_dict = classifier_dict["classes_dict"]
        self.feature_extractor.load_network(classifier_dict["feature_extractor"])
        self.label_classifier.load_network(classifier_dict["label_classifier"])
        # self.load_state_dict(classifier_dict["state_dict"])

    def forward(self, xyz, radius=None):
        x = self.feature_extractor(xyz)
        if radius is not None:
            x = torch.cat([x, radius], dim=1)
        else:
            x = torch.cat([x, x.new_zeros(x.shape[0], 1)], dim=1)
        x = self.label_classifier(x)
        x = F.log_softmax(x, -1)
        return x

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ValueError("name should be a string")
        self._basename = value
        self._name = "{}_Nclasses{}_FeaturesLen{}".format(self._basename, self.num_classes, self.features_len)

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        print("Setting number of classes to {}.".format(value))
        self._num_classes = value
        self.label_classifier.set_output_channel(value)
        self._name = "{}_Nclasses{}_FeaturesLen{}".format(self._basename, self.num_classes, self.features_len)

    @property
    def features_len(self):
        return self._features_len

    @features_len.setter
    def features_len(self, value):
        self._features_len = value
        self.feature_extractor.features_len = value
        self.label_classifier.in_channel = value + 1
        self._name = "{}_Nclasses{}_FeaturesLen{}".format(self._basename, self.num_classes, self.features_len)