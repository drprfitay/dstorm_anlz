import os
import shutil
from pandas.io.parsers import read_csv
import torch
import numpy as np
from copy import deepcopy

from utils.torch_pointcloud_utils import xyz_normalize, sample_and_group
from utils.MiscUtils import Parser

class DatasetBuilder:
    def __init__(self):
        self.parser = Parser()

    class Config:
        def __init__(self):
            self.train_data = None
            self.classes = {}
            self.dataset_output = None
            self.probe = 0
            self.oversample = 0
            self.npoints = None
            self.points_vector = ['probe', 'x', 'y', 'z']

    @staticmethod
    def run_from_config(config_path):
        print(f"DatasetBuilder has been called")
        builder = DatasetBuilder()
        config = builder.Config()
        for name in builder.parser.json_parser(config, config_path):
            print(f"Train {name}")
            builder.build(
                train_data=config.train_data,
                dataset_output=config.dataset_output,
                classes=config.classes,
                probe=config.probe,
                oversample=config.oversample,
                npoints=config.npoints,
                points_vector=config.points_vector
            )

    @staticmethod
    def partition_pointcloud(pointcloud, npoints, min_nsamples=0):
        pointcloud = deepcopy(pointcloud)
        pointcloud_dicts = []

        xyz_norm, bounding_radius, bounding_centeroid = xyz_normalize(pointcloud[:, 0:3])
        pointcloud[:, 0:3] = xyz_norm
        pointcloud = pointcloud.unsqueeze(0)  # sample_and_group works on batches
        nsamples = max(min_nsamples, round(pointcloud.size(1) / npoints) * 2)

        if (min_nsamples != 0) and nsamples > min_nsamples:
            print("Number of samples {} exceeds minimum to oversample".format(nsamples))

        if pointcloud.shape[2] == 3:
            new_xyz, new_points = sample_and_group(nsamples, 0.4, npoints, pointcloud, None, 1)
        else:
            new_xyz, new_points = sample_and_group(nsamples, 0.4, npoints,
                                                   pointcloud[:, :, 0:3], pointcloud[:, :, 3:], 1)
        new_points = new_points.squeeze(0)

        for ind, pc in enumerate(new_points, 0):
            pc, partition_radius, partition_centeroid = xyz_normalize(pc)
            pointcloud_dict = {
                'bounding_centeroid': bounding_centeroid,
                'bounding_radius': bounding_radius,
                'partition_centeroid': partition_centeroid,
                'partition_radius': partition_radius,
                'partition_pointcloud': pc,
            }
            pointcloud_dicts.append(pointcloud_dict)

        return pointcloud_dicts, nsamples

    @staticmethod
    def check_points_vector(points_vector):
        try:
            points_vector.remove('probe')
        except ValueError:
            pass
        points_vector = ['probe'] + points_vector

        if points_vector[:4] == ['probe', 'x', 'y', 'z']:
            if len(points_vector) == 4:
                print("Points contains only [x,y,z], no extra features")
            else:
                print("Points contains extra features: {}".format(points_vector[4:]))
        else:
            raise ValueError("points_vector should at least contain ('probe', x, y, z)")

        return points_vector

    def build(self, train_data, dataset_output, classes={},
              probe=0, oversample=0, npoints=1024, points_vector=['x', 'y', 'z']):

        points_vector = self.check_points_vector(points_vector)

        try:
            shutil.rmtree(dataset_output)
        except FileNotFoundError:
            pass
        os.makedirs(dataset_output, exist_ok=True)

        print('Creating dataset at {}'.format(dataset_output))
        # max_xy = np.zeros(2)
        # print(f'max xy: {max_xy}')
        nfiles_dict = {}
        nfiles_max = 0
        for item in classes:
            dir_point = os.path.join(train_data, item)
            fns = os.listdir(dir_point)
            nfiles_dict[item] = len(list(filter(lambda x: x == '.csv', [os.path.splitext(i)[1] for i in fns])))
            print("Number of files in {}: {}".format(item, nfiles_dict[item]))
            nfiles_max = max(nfiles_max, nfiles_dict[item])

            for fn in fns:
                file_name, file_extension = os.path.splitext(fn)
                if file_extension == ".csv":
                    with open(os.path.join(dir_point, fn), 'r') as f:
                        csv_dict = read_csv(f, usecols=points_vector)[points_vector]
                        # if csv_dict.x.max() > max_xy[0]:
                        #     max_xy[0] = csv_dict.x.max()
                        #     print(f'max xy: {max_xy}')
                        # if csv_dict.y.max() > max_xy[1]:
                        #     max_xy[1] = csv_dict.y.max()
                        #     print(f'max xy: {max_xy}')
        # normalize = np.sqrt(np.sum(max_xy*max_xy))

        for item in classes:
            outdir_point = os.path.join(dataset_output, item)
            os.mkdir(outdir_point)
            indir_point = os.path.join(train_data, item)
            fns = os.listdir(indir_point)

            nexamples = 0
            if oversample != 0:
                min_nsamples = round(oversample * nfiles_max / nfiles_dict[item])
                print("Working on class {}: min {} samples per example".format(item, min_nsamples))
            else:
                min_nsamples = 0
                print("Working on class {}".format(item))

            for fn in fns:
                file_name, file_extension = os.path.splitext(fn)
                if file_extension == ".csv":
                    nexamples += 1
                    with open(os.path.join(indir_point, fn), 'r') as f:
                        csv_dict = read_csv(f, usecols=points_vector)[points_vector]
                        csv_dict = csv_dict[csv_dict['probe'] == probe]
                        if csv_dict.empty:
                            print("{} is empty for probe {}".format(file_name, probe))
                            continue
                        del csv_dict['probe']
                        pointcloud = torch.FloatTensor(csv_dict.values)
                        pointcloud_dicts = self.partition_pointcloud(pointcloud, npoints, min_nsamples)
                        for ind, pc in enumerate(pointcloud_dicts):
                            torch.save(pc, os.path.join(outdir_point, file_name + f'_{ind}.pts'))

            print("Class {}: Number of examples processed {} ; Number of samples created {}\n"
                  .format(item, nexamples, len(os.listdir(outdir_point))))

        with open(os.path.join(dataset_output, "classes.txt"), 'w') as f:
            for k, v in classes.items():
                f.write(f'{k} {v}\n')


if __name__ == '__main__':
    # config = '/home/oronlevy/ext/config/build/edens_a53t_2048_build_config.json'
    # DatasetBuilder.run_from_config(config)

    builder = DatasetBuilder()
    builder.build(
        train_data='/home/oronlevy/ext/data/train_data/Noas_10test_train_flat',
        dataset_output='/home/oronlevy/ext/data/training_datasets/noas_1024_photon_count',
        classes={'Naive': 0, 'TBB': 1},
        probe=0,
        oversample=10,
        npoints=512,
        points_vector=['x', 'y', 'z', 'photon-count']
    )



