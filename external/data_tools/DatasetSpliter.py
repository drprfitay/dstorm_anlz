import os
import math
import json

class DatasetSpliter:
    _persist_methods = ['print_to_log', 'root_training_datasets_dir', 'random_shuffle']

    def __init__(self, setup):
        self._setup = setup

        self._train_split = 80
        self._remainder = None

    def __getattr__(self, attribute):
        if attribute in self._persist_methods:
            return getattr(self._setup, attribute)
        raise AttributeError("{} doesn't have attribute {}".format(self.__module__, attribute))

    def set_remainder_to_none(self):
        self.remainder = None

    def set_remainder_to_train(self):
        self.remainder = 'train'

    def set_remainder_to_valid(self):
        self.remainder = 'valid'

    def get_classes_dict(self, dataset):
        dataset = os.path.abspath(os.path.join(self.root_training_datasets_dir, dataset))
        clsfile = os.path.join(dataset, 'classes.txt')
        cat = {}
        ind = 0
        with open(clsfile, 'r') as f:
            for line in f:
                cat[ind] = line.strip()
                ind += 1

        return cat

    def __call__(self, dataset, classes=None):
        dataset = os.path.abspath(os.path.join(self.root_training_datasets_dir, dataset))
        self.print_to_log('Creating split file at {}'.format(dataset))

        clsfile = os.path.join(dataset, 'classes.txt')
        cat = {}
        ind = 0
        with open(clsfile, 'r') as f:
            for line in f:
                if (classes is None) or (line.strip() in classes):
                    cat[ind] = line.strip()
                    ind += 1

        self.print_to_log("Class labeling")
        train_classes = []
        min_class = ''
        min_samples = 1e6
        for ind in cat.keys():
            self.print_to_log("{}\t{}".format(ind, cat[ind]))
            train_classes.append("{}\t{}".format(ind, cat[ind]))

            dir_point = os.path.join(dataset, cat[ind])
            nsamples = len(os.listdir(dir_point))
            if nsamples < min_samples:
                min_samples = nsamples
                min_class = cat[ind]

        train_size = math.floor(self.train_split / 100.0 * min_samples)
        self.print_to_log("Number of classes to train on: %d" % (len(cat.keys())))
        self.print_to_log("Sparsest class is %s with %d samples" % (min_class, min_samples))
        self.print_to_log("Training / Validation samples per class: %d / %d" % (train_size, min_samples - train_size))

        with open(os.path.join(dataset, "train_classes.txt"), 'w') as f:
            f.write("\n".join(train_classes))

        out_dict = {}
        out_dict['train'] = []
        out_dict['valid'] = []
        for cls in cat.keys():
            dir_point = os.path.join(dataset, cat[cls])
            fns = os.listdir(dir_point)
            self.random_shuffle(fns)

            total_remainder = 0
            for ind, fn in enumerate(fns, 0):
                if ind < train_size:
                    out_dict['train'].append((cls, os.path.abspath(os.path.join(dir_point, fn))))
                elif ind < min_samples:
                    out_dict['valid'].append((cls, os.path.abspath(os.path.join(dir_point, fn))))
                elif self.remainder is not None:
                    total_remainder += 1
                    out_dict[self.remainder].append((cls, os.path.abspath(os.path.join(dir_point, fn))))

            if total_remainder != 0:
                self.print_to_log("Remainder {} samples in class {} went to {}"
                                  .format(total_remainder, cat[cls], self.remainder))

        self.print_to_log("Total Training / Validation samples: {} / {}"
                          .format(len(out_dict['train']), len(out_dict['valid'])))

        with open(os.path.join(dataset, 'split.json'), 'w') as fp:
            json.dump(out_dict, fp)

    @property
    def train_split(self):
        return self._train_split

    @train_split.setter
    def train_split(self, value):
        if value <= 0 or value >= 100:
            raise ValueError("Split must be an integer between 1-99")
        self.print_to_log("Setting train_split to {}".format(value))
        self._train_split = value

    @property
    def remainder(self):
        return self._remainder

    @remainder.setter
    def remainder(self, value):
        if isinstance(value, str):
            if value.lower() in ['train', 'valid']:
                self.print_to_log("Setting remainder to {}".format(value.lower()))
                self._remainder = value.lower()
                return
        self.print_to_log("Setting remainder to None")
        self._remainder = None