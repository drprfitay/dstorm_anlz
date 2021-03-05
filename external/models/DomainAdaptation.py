import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import copy

from models.PointNet import PointNet2
from models.Classifiers import FCNClassifier

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class DAClassifier(nn.Module):
    # _persist_methods = ['print_to_log']

    def __init__(self, setup, features_len=1024, num_classes=3, num_domains=2):
        super(DAClassifier, self).__init__()

        self._setup = setup
        self.print_to_log = self._setup.print_to_log

        self._single_label_classifier = True
        self._features_len = features_len
        self._num_domains = num_domains
        self._num_classes = num_classes

        # train params
        self.train_dataset = []
        self.classes_dict = []
        self.npoint_input = None
        self.domain_criterion = nn.NLLLoss()
        self.label_criterion = nn.NLLLoss()

        # eval params
        self.classifier_output_head = None

        self._basename = "DA_PointNetClassifier"
        self._name = "{}_Nclasses{}_Ndomains{}_FeaturesLen{}"\
            .format(self._basename, self.num_classes, self.num_domains, self.features_len)

        self.feature_extractor = PointNet2(self._setup, self.features_len)

        # Domain Discriminator:
        # 3 layers classification network: input features from pointcloud + radius ; output number of domains
        self.gradient_reversal = GradientReversal(lambda_=1)
        self.domain_discriminator = FCNClassifier(self._setup, self.features_len + 1, [512, 256, self.num_domains])
        self.domain_discriminator.name = 'Domain Discriminator'

        # Label Classifier:
        # 3 layers classification network: input features from pointcloud + radius ; output number of classes
        self.label_classifier = FCNClassifier(self._setup, self.features_len + 1, [512, 256, self.num_classes])
        self.label_classifier.name = 'Label Classifier'

    # def __getattr__(self, attribute):
    #     if attribute in self._persist_methods:
    #         return getattr(self._setup, attribute)
    #     raise AttributeError("{} doesn't have attribute {}".format(self.__module__, attribute))


    def load_feature_extractor(self, feature_extractor_dict):
        self.feature_extractor.load_network(feature_extractor_dict)

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def export_network(self):
        # self.print_to_log("Exporting DA PointNet Classifier {}".format(self.name))
        classifier_dict = {}
        classifier_dict["name"] = self.name
        classifier_dict["basename"] = self._basename
        classifier_dict["features_length"] = self.features_len
        classifier_dict["num_domains"] = self.num_domains
        classifier_dict["train_dataset"] = self.train_dataset
        classifier_dict["npoint_input"] = self.npoint_input
        classifier_dict["classes_dict"] = self.classes_dict
        classifier_dict["classifier_output_head"] = self.classifier_output_head
        classifier_dict["feature_extractor"] = self.feature_extractor.export_network()
        classifier_dict["domain_discriminator"] = self.domain_discriminator.export_network()

        classifier_dict["single_label_classifier"] = self.single_label_classifier
        classifier_dict["num_classes"] = self.num_classes
        if self.single_label_classifier:
            classifier_dict["label_classifier"] = self.label_classifier.export_network()
        else:
            classifier_dict["label_classifier"] = []
            for label_classifier in self.label_classifier:
                classifier_dict["label_classifier"].append(label_classifier.export_network())

        # classifier_dict["state_dict"] = copy.deepcopy(self.state_dict())
        return classifier_dict

    def load_network(self, classifier_dict):
        self.print_to_log("Loading DA PointNet Classifier {}".format(self.name))
        self._name = classifier_dict["name"]
        self._basename = classifier_dict["basename"]
        self._features_len = classifier_dict["features_length"]
        self._num_domains = classifier_dict["num_domains"]
        self.train_dataset = classifier_dict["train_dataset"]
        self.npoint_input = classifier_dict["npoint_input"]
        self.classes_dict = classifier_dict["classes_dict"]
        self.classifier_output_head = classifier_dict["classifier_output_head"]
        self.feature_extractor.load_network(classifier_dict["feature_extractor"])
        self.domain_discriminator.load_network(classifier_dict["domain_discriminator"])

        if classifier_dict["single_label_classifier"]:
            self.set_single_label_classifier(classifier_dict["num_classes"])
            self.label_classifier.load_network(classifier_dict["label_classifier"])
        else:
            self.set_multiple_label_classifiers(classifier_dict["num_classes"])
            for ind, label_classifier in enumerate(self.label_classifier, 0):
                label_classifier.load_network(classifier_dict["label_classifier"][ind])

        # self.load_state_dict(classifier_dict["state_dict"])

    def forward(self, xyz, radius=None):
        x = self.feature_extractor(xyz)
        if radius is not None:
            x = torch.cat([x, radius], dim=1)
        else:
            x = torch.cat([x, x.new_zeros(x.shape[0], 1)], dim=1)

        if self.single_label_classifier:
            labels = self.label_classifier(x)
            labels = F.log_softmax(labels, -1)
        else:
            labels = []
            for label_classifier in self.label_classifier:
                y = label_classifier(x)
                labels.append(F.log_softmax(y, -1))

        y = self.gradient_reversal(x)
        domains = self.domain_discriminator(y)
        domains = F.log_softmax(domains, -1)

        if self.classifier_output_head is not None:
            if self.single_label_classifier:
                return labels
            else:
                return labels[self.classifier_output_head]
        else:
            return labels, domains

    def set_multiple_label_classifiers(self, num_classes_list):
        self._single_label_classifier = False
        self._num_classes = num_classes_list
        self.label_classifier = nn.ModuleList()
        self.num_domains = len(num_classes_list)
        for ind, num_classes in enumerate(num_classes_list, 0):
            self.label_classifier.append(FCNClassifier(self._setup, self.features_len + 1, [512, 256, num_classes]))
            self.label_classifier[-1].name = 'Label Classifier (Domain {})'.format(ind)
        self.name = self._basename

    def set_single_label_classifier(self, num_classes):
        self._single_label_classifier = True
        self._num_classes = num_classes
        self.label_classifier = FCNClassifier(self._setup, self.features_len + 1, [512, 256, self.num_classes])
        self.name = self._basename

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ValueError("name should be a string")
        self._basename = value
        if self.single_label_classifier:
            self._name = "{}_Nclasses{}_Ndomains{}_FeaturesLen{}"\
                .format(self._basename, self.num_classes, self.num_domains, self.features_len)
        else:
            self._name = "{}_MultiHead_Ndomains{}_FeaturesLen{}"\
                .format(self._basename, self.num_domains, self.features_len)


    @property
    def single_label_classifier(self):
        return self._single_label_classifier

    @property
    def num_classes(self):
        if self.classifier_output_head is not None:
            if self.single_label_classifier:
                return self._num_classes
            else:
                return self._num_classes[self.classifier_output_head]
        else:
            return self._num_classes

    @property
    def num_domains(self):
        return self._num_domains

    @num_domains.setter
    def num_domains(self, value):
        if (not self.single_label_classifier) and (len(self.num_classes) != value):
            raise ValueError("In multiple label classifier mode, you can't set number of domains different "
                             "from number of label classifiers")
        self._num_domains = value
        self.domain_discriminator.num_classes = value
        self.name = self._basename

    @property
    def features_len(self):
        return self._features_len

    @features_len.setter
    def features_len(self, value):
        self._features_len = value
        self.feature_extractor.features_len = value
        self.domain_discriminator.features_len = value
        if self.single_label_classifier:
            self.label_classifier.features_len = value
        else:
            for i in len(self.num_classes):
                self.label_classifier[i].features_len = value
        self.name = self._basename