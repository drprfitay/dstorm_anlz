
from __future__ import print_function
import os
from parse_single import main
import re



#-----------------------FILL__INFO------------------------------------------#

data_type = "2d"
epsilon = 100
minimum_neighbors = 10

main_folder = "D:/Gilad/"    # where all the sub_folders are at
direrctories = []
for root, dirs, files in os.walk(main_folder, topdown=True):
    for name in dirs:
        direrctories.append(os.path.join(root, name))
print(direrctories)

filess = []
for director in direrctories:
    current_dir = director
    for root, dirs, files in os.walk(director, topdown=True):
        green_filess = []
        red_filess = []
        for name in files:
            if re.findall(r".*?green.csv", name, re.DOTALL):
                green_filess.append(os.path.join(root, name))
            if re.findall(r".*?red.csv", name, re.DOTALL):
                red_filess.append(os.path.join(root, name))
        if len(green_filess) > 0:
            for green in green_filess:
                for red in red_filess:
                    g = green.find('green')
                    r = red.find('red')
                    green_str = green[:g]
                    red_str = green[:r]
                    if green_str == red_str:
                        run_clust(current_dir, green, red, epsilon, minimum_neighbors, data_type)

