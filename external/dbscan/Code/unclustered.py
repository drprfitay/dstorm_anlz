__author__ = 'UriA12'

import os
from classes import *
from parse_main import main as parse_main
import time

DEBUG = False


def main(sample, data_type, mini_eps, mini_minimum_neighbors, file_direc, mode=2, color="both", dummy_file="", unclustered_by_filter_points=None):
    file_dir = file_direc + "/unclus_analysis_{}".format(time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))
    os.mkdir(file_dir)

    if unclustered_by_filter_points == None:
        unclustered_points = sample.unclustered_points + get_unclustered_by_filter(sample.red_clusters,
                                                                               sample.green_clusters)
    else:
        unclustered_points = sample.unclustered_points + unclustered_by_filter_points

    red_points = [point for point in unclustered_points if point.color == "red"]
    green_points = [point for point in unclustered_points if point.color == "green"]
    new_green_name = file_dir + "/new_green_points.csv"
    new_red_name = file_dir + "/new_red_points.csv"
    titles = "null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,x,y,z,null,null\n"
    line = "null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,{},{},{},null,null\n"

    # Create files with coordinates:
    with open(new_green_name, "w") as f:
        f.write(titles)
        for green in green_points:
            f.write(line.format(green.point[0], green.point[1], green.point[2]))
    with open(new_red_name, "w") as f:
        f.write(titles)
        for red in red_points:
            f.write(line.format(red.point[0], red.point[1], red.point[2]))

    # Run main again:
    is_3d = True if (data_type == "new_3d") or (data_type == "3d") else False
    data_type = "3d" if is_3d else "2d"  # careful here...
    if mode == 1 and color == "red":
        new_green_name = dummy_file
    if mode == 1 and color == "green":
        new_red_name = dummy_file
    parse_main(data_type, mini_eps, mini_minimum_neighbors, mini_eps, mini_minimum_neighbors, new_green_name,
               new_red_name, "unclust_p2", file_dir, mode)


def get_unclustered_by_filter(red_clusters, green_clusters, color_a="both", red_points="MIN;MAX",
                              green_points="MIN;MAX", density="0.0098;MAX", size="MIN;MAX"):
    res = []

    MAX = 1000000.
    MIN = -1.

    color = color_a

    red_points_lst = red_points.split(";")
    red_points_min = MIN if red_points_lst[0] == "MIN" else float(red_points_lst[0])
    red_points_max = MAX if red_points_lst[1] == "MAX" else float(red_points_lst[1])

    if DEBUG: print(green_points)
    green_points_lst = green_points.split(";")
    green_points_min = MIN if green_points_lst[0] == "MIN" else float(green_points_lst[0])
    green_points_max = MAX if green_points_lst[1] == "MAX" else float(green_points_lst[1])

    if DEBUG: print(density)
    density_lst = density.split(";")
    density_min = MIN if density_lst[0] == "MIN" else float(density_lst[0])
    density_max = MAX if density_lst[1] == "MAX" else float(density_lst[1])
    if DEBUG: print(size)

    size_lst = size.split(";")
    size_min = MIN if size_lst[0] == "MIN" else float(size_lst[0])
    size_max = MAX if size_lst[1] == "MAX" else float(size_lst[1])

    for cluster in red_clusters:
        try:
            if color != "both":
                if "red" != color: res += cluster.points
            elif float(len(cluster.points)) > red_points_max or float(len(cluster.points)) < red_points_min:
                res += cluster.points
            elif len(cluster.points) * 10000 / ((float(cluster.size) ** 3) * (4 / 3) * math.pi) > density_max or len(
                    cluster.points) * 10000 / ((float(cluster.size) ** 3) * (4 / 3) * math.pi) < density_min:
                res += cluster.points
            elif float(cluster.size) > size_max or float(cluster.size) < size_min:
                res += cluster.points
        except ZeroDivisionError:
            pass
    for cluster in green_clusters:
        try:
            if color != "both":
                if "green" != color: res += cluster.points
            elif float(len(cluster.points)) > green_points_max or float(len(cluster.points)) < green_points_min:
                res += cluster.points
            elif len(cluster.points) * 10000 / ((float(cluster.size) ** 3) * (4 / 3) * math.pi) > density_max or len(
                    cluster.points) * 10000 / ((float(cluster.size) ** 3) * (4 / 3) * math.pi) < density_min:
                res += cluster.points
            elif float(cluster.size) > size_max or float(cluster.size) < size_min:
                res += cluster.points
        except ZeroDivisionError:
            pass
    return res


def get_unclustered_by_filter_main(red_clusters, green_clusters, color_a="both", points="MIN;MAX", red_points="MIN;MAX",
                                   green_points="MIN;MAX", density="0.0098;MAX", colocalization="false",
                                   anglex="MIN;MAX", angley="MIN;MAX", size="MIN;MAX"):
    res = []

    MAX = 1000000.
    MIN = -1.

    color = color_a

    points_lst = points.split(";")
    points_min = MIN if points_lst[0] == "MIN" else float(points_lst[0])
    points_max = MAX if points_lst[1] == "MAX" else float(points_lst[1])

    red_points_lst = red_points.split(";")
    red_points_min = MIN if red_points_lst[0] == "MIN" else float(red_points_lst[0])
    red_points_max = MAX if red_points_lst[1] == "MAX" else float(red_points_lst[1])

    if DEBUG: print(green_points)
    green_points_lst = green_points.split(";")
    green_points_min = MIN if green_points_lst[0] == "MIN" else float(green_points_lst[0])
    green_points_max = MAX if green_points_lst[1] == "MAX" else float(green_points_lst[1])

    if DEBUG: print(density)
    density_lst = density.split(";")
    density_min = MIN if density_lst[0] == "MIN" else float(density_lst[0])
    density_max = MAX if density_lst[1] == "MAX" else float(density_lst[1])
    if DEBUG: print(size)

    size_lst = size.split(";")
    size_min = MIN if size_lst[0] == "MIN" else float(size_lst[0])
    size_max = MAX if size_lst[1] == "MAX" else float(size_lst[1])

    anglex_lst = anglex.split(";")
    anglex_min = MIN if anglex_lst[0] == "MIN" else float(anglex_lst[0])
    anglex_max = MAX if anglex_lst[1] == "MAX" else float(anglex_lst[1])

    angley_lst = angley.split(";")
    angley_min = MIN if angley_lst[0] == "MIN" else float(angley_lst[0])
    angley_max = MAX if angley_lst[1] == "MAX" else float(angley_lst[1])

    coloc = False if colocalization == "no" else True

    for cluster in red_clusters:
        try:
            if color != "both":
                if "red" != color:
                    res += cluster.points
            elif float(len(cluster.points)) > points_max or float(len(cluster.points)) < points_min:
                res += cluster.points
            elif float(len(cluster.points)) > red_points_max or float(len(cluster.points)) < red_points_min:
                res += cluster.points
            elif len(cluster.points) * 10000 / ((float(cluster.size) ** 3) * (4 / 3) * math.pi) > density_max or len(
                    cluster.points) * 10000 / ((float(cluster.size) ** 3) * (4 / 3) * math.pi) < density_min:
                res += cluster.points
            elif float(cluster.size) > size_max or float(cluster.size) < size_min:
                res += cluster.points
            elif colocalization != "all":
                if not coloc == cluster.is_colocalized:  # (coloc and not cluster.is_colocalized) or (not coloc and cluster.is_colocalized):
                    res += cluster.points
            elif cluster.angle_x > anglex_max or cluster.angle_y > angley_max or cluster.angle_x < anglex_min or cluster.angle_y < angley_min:
                res += cluster.points
        except ZeroDivisionError:
            pass

    for cluster in green_clusters:
        try:
            if color != "both":
                if "green" != color:
                    res += cluster.points
            elif float(len(cluster.points)) > points_max or float(len(cluster.points)) < points_min:
                res += cluster.points
            elif float(len(cluster.points)) > green_points_max or float(len(cluster.points)) < green_points_min:
                res += cluster.points
            elif len(cluster.points) * 10000 / ((float(cluster.size) ** 3) * (4 / 3) * math.pi) > density_max or len(
                    cluster.points) * 10000 / ((float(cluster.size) ** 3) * (4 / 3) * math.pi) < density_min:
                res += cluster.points
            elif float(cluster.size) > size_max or float(cluster.size) < size_min:
                res += cluster.points
            elif not coloc == cluster.is_colocalized:  # (coloc and not cluster.is_colocalized) or (not coloc and cluster.is_colocalized):
                res += cluster.points
            elif cluster.angle_x > anglex_max or cluster.angle_y > angley_max or cluster.angle_x < anglex_min or cluster.angle_y < angley_min:
                res += cluster.points
        except ZeroDivisionError:
            pass
    return res
