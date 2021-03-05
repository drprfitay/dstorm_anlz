from torch.utils.data import Sampler
import random


class NoasGroupSampler(Sampler):
    def __new__(cls, data_source, rows=None, ratio=1, coprotein=None, zone=None, *args, **kwargs):
        if rows is not None:
            train_sampler = super(NoasGroupSampler, cls).__new__(cls)
            train_sampler.__init__(data_source, rows)
            return train_sampler

        df = data_source.df
        if coprotein is not None:
            df = df.query(f"coprotein in {coprotein}")
        if zone is not None:
            df = df.query(f"zone in {zone}")
        rows = df.index.values.tolist()

        if ratio == 1:
            train_sampler = super(NoasGroupSampler, cls).__new__(cls)
            train_sampler.__init__(data_source, rows)
            return train_sampler
        elif 0 < ratio < 1:
            middle_point = int(ratio*len(rows))
            random.shuffle(rows)
            train_sampler = super(NoasGroupSampler, cls).__new__(cls)
            train_sampler.__init__(data_source, rows[:middle_point])
            valid_sampler = super(NoasGroupSampler, cls).__new__(cls)
            valid_sampler.__init__(data_source, rows[middle_point:])
            return train_sampler, valid_sampler
        else:
            raise ValueError("ratio must be within (0,1]")

    def __init__(self, data_source, rows=None, *args, **kwargs):
        super(NoasGroupSampler, self).__init__(data_source)
        self.data_source = data_source

        if rows is not None:
            self.indices_list = rows
        else:
            self.indices_list = list(range(len(self.data_source)))

        random.shuffle(self.indices_list)

    def __iter__(self):
        random.shuffle(self.indices_list)
        return iter(self.indices_list)

    def __len__(self):
        return len(self.indices_list)


class MinSampler(Sampler):

    def __init__(self, df, classes_list=None, exclude_indices=None):
        self.indices_list = []

        if classes_list is not None:
            df = df.query(f'label_class in {classes_list}')

        if exclude_indices is not None:
            df = df.drop(exclude_indices, axis=0)

        self.classes = dict(zip(df.label_class.unique().tolist(), df.label.unique().tolist()))
        self.num_classes = len(self.classes.keys())
        self.min_samples = df.label.value_counts().min()

        for label_class in self.classes:
            indices = df.query(f'label == {self.classes[label_class]}').index.values
            print(f'{label_class} has {len(indices)} samples')
            self.indices_list.extend(random.choices(indices, k=self.min_samples))

        random.shuffle(self.indices_list)
        print(f'Min number of samples to select from {self.min_samples}')
        print(f'Number of training samples: {self.num_classes}x{self.min_samples}={len(self.indices_list)}')

    def __iter__(self):
        return iter(self.indices_list)

    def __len__(self):
        return len(self.indices_list)


class MinReductionSampler(Sampler):

    def __init__(self, df, reduction_rate=0.8, classes_list=None):
        self.indices_list = []

        if classes_list is not None:
            df = df.query(f'label_class in {classes_list}')

        self.classes = dict(zip(df.label_class.unique().tolist(), df.label.unique().tolist()))
        self.num_classes = len(self.classes.keys())
        self.min_samples = df.label.value_counts().min()
        self.num_samples = int(reduction_rate * self.min_samples)

        for label_class in self.classes:
            indices = df.query(f'label == {self.classes[label_class]}').index.values
            print(f'{label_class} has {len(indices)} samples')
            self.indices_list.extend(random.choices(indices, k=self.num_samples))

        random.shuffle(self.indices_list)
        print(f'Min number of samples to select from {self.min_samples}')
        print(f'Number of training samples: {self.num_classes}x{self.num_samples}={len(self.indices_list)}')

    def __iter__(self):
        return iter(self.indices_list)

    def __len__(self):
        return len(self.indices_list)
