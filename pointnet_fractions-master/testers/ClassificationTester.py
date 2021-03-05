import os
from pandas.io.parsers import read_csv
import torch

from utils.MiscUtils import Parser
from utils.torch_pointcloud_utils import xyz_normalize
from data_tools.DatasetBuilder import DatasetBuilder

Blue = lambda x: '\033[94m' + x + '\033[0m'


class ClassificationTester:

    def __init__(self):
        self.parser = Parser()
        self.partition_pointcloud = DatasetBuilder.partition_pointcloud
        self.check_points_vector = DatasetBuilder.check_points_vector

    class Config:
        def __init__(self):
            self.probe = 0
            self.points_vector = ['probe', 'x', 'y', 'z']
            self.partition = True  # If True - partition to npoint clouds, False - All the pointcloud as input
            self.min_nsamples = 0 # relevant only with partitioning
            self.npoints = 0  # relevant only with partitioning - if '0', take npoint from trained model

    def run_from_config(self, config_path, classifier, points_vector=['x', 'y', 'z'], device=None, output_dir=None):
        print(f"Classification Tester has been called")
        config = self.Config()
        for name in self.parser.json_parser(config, config_path):
            print(f"Test {name}")
            self.test(
                classifier=classifier,
                test_dir=config.test_dir,
                device=device,
                points_vector=points_vector,
                probe=config.probe,
                partition=config.partition,
                min_nsamples=config.min_nsamples,
                npoints=config.npoints,
                output_dir=output_dir
            )

    def test(self, classifier, test_dir, device=None,
              points_vector=['x', 'y', 'z'], probe=0, partition=False, min_nsamples=0, npoints=None, output_dir=None):

        results = {
            "classifier": classifier.name,
            "test_dir": test_dir,
            "points_vector": points_vector,
            "probe": probe,
            "partition": partition,
            "min_nsamples": min_nsamples,
            "npoints": npoints,
            "output_dir": output_dir,
            "raw_stats": {}
        }
        points_vector = self.check_points_vector(points_vector)

        if npoints is None:
            npoints = int(classifier.npoint_input)

        num_classes = classifier.num_classes

        print("Test Classifier: {}. Test dir: {}".format(classifier.name, test_dir))
        if isinstance(classifier.classes_dict, list):
            print("Classifier number of classes is {}:".format(num_classes))
            for dict in classifier.classes_dict:
                print("\t{}:".format(dict))
        else:
            print("Classifier number of classes is {}: {}".format(num_classes, classifier.classes_dict))

        device = device if device is not None else torch.device("cpu")
        classifier.to(device)
        classifier.eval()

        for child in os.listdir(test_dir):
            test_path = os.path.join(test_dir, child)
            if os.path.isdir(test_path):
                label_name = os.path.basename(test_path)
                print(f"'--------{label_name}---------'")
                fns = os.listdir(test_path)

                total = 0
                log_softmax_count = num_classes * [0]
                bin_counts_count = num_classes * [0]
                predictions = torch.LongTensor(num_classes).zero_()
                for fn in fns:
                    file_name, file_extansion = os.path.splitext(fn)
                    if file_extansion == ".csv":
                        with open(os.path.join(test_path, fn), 'r') as f:
                            csv_dict = read_csv(f, usecols=points_vector)[points_vector]
                            csv_dict = csv_dict[csv_dict['probe'] == probe]
                            if csv_dict.empty:
                                print("{} is empty for probe {}".format(file_name, probe))
                                continue
                            del csv_dict['probe']

                        pointcloud = torch.FloatTensor(csv_dict.values)

                        if partition:
                            pointcloud = torch.FloatTensor(csv_dict.values)
                            pointcloud_dicts, nsamples = self.partition_pointcloud(pointcloud, npoints, min_nsamples)
                            print("{}: number of points per partition {} ; number of partitions: {}"
                                  .format(file_name, npoints, nsamples))

                            pointcloud = torch.FloatTensor()
                            radius = torch.FloatTensor()
                            for pc in pointcloud_dicts:
                                pointcloud = torch.cat((pointcloud, pc['partition_pointcloud'].unsqueeze(0)), dim=0)
                                radius = torch.cat((radius, pc['partition_radius'].unsqueeze(0)), dim=0)
                            pc_batch_split = pointcloud.split(8, dim=0)
                            rad_batch_split = radius.split(8, dim=0)

                            log_softmax = torch.FloatTensor(num_classes).zero_().to(device)
                            pred_choice = torch.LongTensor().to(device)
                            for ind, (pointcloud, radius) in enumerate(zip(pc_batch_split, rad_batch_split)):
                                pointcloud = pointcloud.permute(0, 2, 1)
                                radius = radius.unsqueeze(1)
                                pointcloud, radius = pointcloud.to(device), radius.to(device)

                                pred = classifier(pointcloud, radius)

                                x = (pred > 0).sum()
                                if x > 0:
                                    raise ValueError("Error pred is positive")

                                pred_choice = torch.cat((pred_choice, pred.data.max(1)[1]))

                                log_softmax += pred.sum(0)

                            total += 1

                            log_softmax_max = log_softmax.max(0)[1]
                            str_format = lambda x: '{:.2f}'.format(min(-x, 99.99))
                            log_softmax = list(map(str_format, log_softmax.tolist()))
                            log_softmax[log_softmax_max] = Blue(log_softmax[log_softmax_max])
                            log_softmax_count[log_softmax_max] += 1

                            bin_counts = torch.bincount(pred_choice, minlength=num_classes)
                            bin_counts_max = bin_counts.max(0)[1]
                            bin_counts = list(map(str, bin_counts.tolist()))
                            bin_counts[bin_counts_max] = Blue(bin_counts[bin_counts_max])
                            bin_counts_count[bin_counts_max] += 1

                            log_softmax_template = ["{:^8}"] * num_classes
                            log_softmax_template[log_softmax_max] = Blue(log_softmax_template[log_softmax_max])
                            log_softmax_template = '|'.join(log_softmax_template)
                            bin_counts_template = ["{:^8}"] * num_classes
                            bin_counts_template[bin_counts_max] = Blue(bin_counts_template[bin_counts_max])
                            bin_counts_template = '|'.join(bin_counts_template)
                            print(log_softmax_template.format(*tuple(log_softmax)))
                            print(bin_counts_template.format(*tuple(bin_counts)))
                            print('_' * 15)

                        else:
                            xyz_norm, r1 = xyz_normalize(pointcloud[:, 0:3])
                            pointcloud[:, 0:3] = xyz_norm
                            pointcloud = pointcloud.unsqueeze(0)

                            pointcloud = pointcloud.permute(0, 2, 1)
                            pointcloud = pointcloud.to(device)

                            pred = classifier(pointcloud)

                            pred_choice = torch.LongTensor(num_classes).zero_()
                            pred_choice[pred.argmax()] = 1

                            total += 1
                            predictions += pred_choice

                print("Total number of samples evaluated: {}".format(total))
                if partition:
                    results["raw_stats"][label_name] = {
                        "log_softmax": log_softmax_count,
                        "bin_counts": bin_counts_count
                    }
                    print("log softmax count: {}".format(log_softmax_count))
                    print("majority rule count: {}".format(bin_counts_count))
                else:
                    results["raw_stats"][label_name] = predictions.tolist()
                    print("Decision count: {}".format(predictions.tolist()))

        return results

if __name__ == '__main__':
    from models.Classifiers import PointNetClassifier

    classifier = PointNetClassifier()
    tester = ClassificationTester()

    tester.test(
        classifier=classifier,
        test_dir='/home/oronlevy/ext/data/test_data/Edens_A53T_test_flat/',
        device=None,
        points_vector=['x', 'y', 'z'],
        probe=0,
        partition=True,
        min_nsamples=0,
        npoints=None,
        output_dir=None
    )
