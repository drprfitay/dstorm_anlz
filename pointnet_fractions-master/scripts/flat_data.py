# Projection into Z-plane of dSTORM data format
# python flat_data.py --input-dir ../../../data/parkinson_data --output-dir ../../../data/flat_parkinson_data

import os
import os.path
import shutil
from pandas.io.parsers import read_csv
import argparse

FLAT_AXIS = 'z'

def flat_data(args):

    try:
        shutil.rmtree(args.output_dir)
    except FileNotFoundError:
        pass
    os.mkdir(args.output_dir)

    for subdir, dirs, files in os.walk(args.input_dir):
        relpath = os.path.relpath(subdir, args.input_dir)
        print("Current subdir path {}".format(relpath))

        try:
            os.mkdir(os.path.join(args.output_dir, relpath))
        except FileExistsError:
            pass

        for file in files:
            file_name, file_extansion = os.path.splitext(file)
            if file_extansion == ".csv":
                with open(os.path.join(subdir, file), 'r') as f:
                    csv_dict = read_csv(f)
                    csv_dict['z'] = [0 for i in range(len(csv_dict[FLAT_AXIS]))]
                    csv_dict.to_csv(os.path.join(args.output_dir, relpath, file_name + '.csv'))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for build dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True, help="data input directory")
    parser.add_argument("--output-dir", type=str, required=True, help="output directory")
    parser.add_argument("--npoint", type=int, default=2048, help="number of points per sample")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    flat_data(args)
