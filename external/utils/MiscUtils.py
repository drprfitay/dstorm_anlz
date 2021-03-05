import os
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import json
from pandas import read_json


class Log:
    def __init__(self, log_path=None):

        self.log_path = log_path

        if self.log_path is not None:
            if os.path.exists(self.log_path):
                with open(self.log_path, 'a') as f:
                    f.write(100*'_' + '\n\n')

        self.log_open_datetime = datetime.now()
        self.start_time = datetime.now()

        self.color_dic = {
            'grey'  : '\033[90m',
            'red'   : '\033[91m',
            'green' : '\033[92m',
            'yellow': '\033[93m',
            'blue'  : '\033[94m',
            'violet': '\033[95m',
        }
        self.color_end = '\033[0m'

    def _get_datetime(self, only_date=False, current=False):
        '''
        :param:     current: True/False - Log opening time or current time
        :return:    str : dd:mm:YY HH:MM:SS
        '''

        if only_date:
            datetime_str = "%d_%m_%Y"
        else:
            datetime_str = "%d_%m_%Y_%H_%M_%S"

        if current:
            return datetime.now().strftime(datetime_str)
        else:
            return self.log_open_datetime.strftime(datetime_str)

    def get_open_date(self):
        return self._get_datetime(True, False)

    def get_current_date(self):
        return self._get_datetime(True, True)

    def get_open_datetime(self):
        return self._get_datetime(False, False)

    def get_current_datetime(self):
        return self._get_datetime(False, True)

    def start_timer(self):
        self.start_time = datetime.now()

    def get_time_elapsed(self):
        now = datetime.now()
        duration = now - self.start_time
        duration_in_s = duration.total_seconds()    # Total number of seconds between dates
        hours = divmod(duration_in_s, 3600)         # Get hours
        minutes = divmod(hours[1], 60)              # Use remainder of hours to calc minutes
        seconds = divmod(minutes[1], 1)             # Use remainder of minutes to calc seconds

        return "%d:%d:%d" % (hours[0], minutes[0], seconds[0])

    def color(self, line='', color='blue'):
        return self.color_dic[color] + line + self.color_end

    def print_to_log(self, line='', color=None, show_time=True):
        if show_time:
            if line != '':
                line = datetime.now().strftime("%d/%m/%Y %H:%M:%S: {}".format(line))

        with open(self.log_path, 'a') as f:
            f.write(str(line) + '\n')

        if color is not None:
            line = self.color(str(line), color)

        print(line)


class Plotter:
    def __init__(self, setup):
        self._setup = setup
        self.print_to_log = self._setup.print_to_log
        self._show = False

        self.color_map = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # , 'w']
        self.markers = ['.', '^', 's', 'p', '*', 'D']

        self.default_filename = ''

    def _save_and_show(self, filename):
        outfile = os.path.join(self._setup.output_dir, filename + "_{}.png".format(self._setup.get_current_datetime()))
        self.print_to_log("Saving image {}".format(outfile))
        plt.savefig(outfile)

        if self._show:
            plt.show()

    def plot_loss_functions(self, train_loss, valid_loss, filename=None):
        plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        plt.plot(train_loss, linewidth=3, label="training")
        plt.plot(valid_loss, linewidth=3, label="validation")

        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()

        if not isinstance(filename, str):
            filename = 'loss_functions'
        self._save_and_show(filename)


    def show2d_scatter(self, pointclouds, filename=None):

        if not isinstance(pointclouds, list):
            pointclouds = [pointclouds]

        fig = plt.figure()
        ax = fig.add_subplot()

        i = 0
        x_min, x_max, y_min, y_max = [0]*4
        for pointcloud in pointclouds:
            if len(pointcloud.size()) == 2:
                pointcloud = pointcloud.unsqueeze(0)
            elif len(pointcloud.size()) != 3:
                self.print_to_log("not enough dim-s in pointcloud input, expected 2 or 3 got {}"
                                  .format(len(pointcloud.size())))
                raise RuntimeError("Too few dims")

            if pointcloud.size()[-1] == 3:
                pointcloud = pointcloud[:,:,:2]
            elif pointcloud.size()[-1] != 2:
                self.print_to_log("expected 2-dim vectors got {}".format(pointcloud.size()[-1]))
                raise RuntimeError("Too few dims")

            for pc in range(pointcloud.size()[0]):
                color = i % len(self.color_map)
                marker = math.floor(i/len(self.markers))
                ax.scatter(pointcloud[pc,:,0], pointcloud[pc,:,1],
                             c=self.color_map[color], marker=self.markers[marker])

                i += 1
                x_left, x_right = plt.xlim()
                y_down, y_up = plt.ylim()
                x_min = min(x_min, x_left)
                x_max = max(x_max, x_right)
                y_min = min(y_min, y_down)
                y_max = max(y_max, y_up)

            self.color_map.pop(0)
            i = 0

        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        plt.xlabel('X')
        plt.ylabel('Y')

        if not isinstance(filename, str):
            filename = 'loss_functions'
        self._save_and_show(filename)

    def show3d_scatter(self, pointcloud, filename=None):
        COLOR_MAP = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # , 'w']
        MARKERS = ['.', '^', 's', 'p', '*', 'D']

        if len(pointcloud.size()) == 2:
            pointcloud = pointcloud.unsqueeze(0)
        elif len(pointcloud.size()) != 3:
            self.print_to_log("not enough dim-s in pointcloud input, expected 2 or 3 got {}"
                              .format(len(pointcloud.size())))
            raise

        if pointcloud.size()[-1] != 3:
            self.print_to_log("expected 3-dim vectors got {}".format(pointcloud.size()[-1]))

        fig = plt.figure()
        ax = Axes3D(fig)

        for pc in range(pointcloud.size()[0]):
            color = pc % len(COLOR_MAP)
            marker = math.floor(pc/len(MARKERS))
            ax.scatter(pointcloud[pc,:,0], pointcloud[pc,:,1], pointcloud[pc,:,2],
                         c=COLOR_MAP[color], marker=MARKERS[marker])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if not isinstance(filename, str):
            filename = 'loss_functions'
        self._save_and_show(filename)


class Parser:
    def __init__(self):
        pass

    def _json_config_read(self, json_path):
        if (not isinstance(json_path, str)) or (json_path[-5:] != '.json'):
            raise ValueError("Builder expect a '.json' configuration file")

        print("Reading json configuration file {}".format(json_path))
        with open(json_path, 'r') as f:
            data = f.read()
            json_dict = json.loads(data)

        return json_dict

    def _parse_config_dict_to_self(self, obj, config_dict):
        for k,v in config_dict.items():
            print("\tSetting {} --> {}".format(k, v))
            setattr(obj, k, v)

    def json_parser(self, obj, json_path):
        json_config = self._json_config_read(json_path)
        for key in json_config.keys():
            print(30*"-")
            print("Parsing config dict {}:".format(key))
            self._parse_config_dict_to_self(obj, json_config[key])
            yield key
