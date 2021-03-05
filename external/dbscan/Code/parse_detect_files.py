__author__ = 'UriA12'
import os
import re
import statistics
import math
import numpy as np
from parse_main import main


# main function, get info from "main_with_gui"

def go(eps, min_ngbs, d_type, pth):
    print(eps, min_ngbs, d_type, pth)
    data_type = d_type  # "2d", etc.
    epsilon = eps
    minimum_neighbors = min_ngbs

    main_folder = pth  # where all the sub_folders are at #should be session folder
    session_name = get_name_super(pth)

    particles_filess = []
    cntr = 0
    green_filess = []
    red_filess = []
    raw_green_filess = []
    raw_red_filess = []

    nrm_cntr = 0
    raw_cntr = 0
    prtcls_cntr = 0
    for root, dirs, files in os.walk(main_folder, topdown=True):
        for name in files:
            if re.findall(r".*?green\.csv", name, re.DOTALL):  # changed from 'r".*?green.*?.csv"'
                green_filess.append(os.path.join(root, name))
                nrm_cntr += 1
                print("normal counter: {}".format(nrm_cntr))
            if re.findall(r".*?red\.csv", name, re.DOTALL):  # changed from 'r".*?red.*?.csv"'
                red_filess.append(os.path.join(root, name))
            if re.findall(r".*?green_r[ao]w\.csv", name, re.DOTALL):
                raw_green_filess.append(os.path.join(root, name))
                raw_cntr += 1
                print("raw counter: {}".format(raw_cntr))
            if re.findall(r".*?red_r[ao]w\.csv", name, re.DOTALL):
                raw_red_filess.append(os.path.join(root, name))
            if re.findall(r"particles\.csv", name):
                if not os.path.join(root, name) in final_particles_files:  # NOTICE!!! NOT SUPPORTED YET!
                    particles_filess.append(os.path.join(root, name))
                    prtcls_cntr += 1
                    print("particles counter: {}".format(prtcls_cntr))

    session_data = open(main_folder + '/' + session_name + "_summary.csv", "w")
    session_data.write("test#, avg green in red clusters, std green in red clusters, avg red in green clusters,\
    std red in green clusters,avg red sphericity, std red sphericity, avg green sphericity,std green sphericity,\
    avg red Xangl, std red Xangl, avg red Yangl, std red Yangl, avg green Xangl, std green Xangl, avg green Yangl,\
    std green Yangl, avg red size, std red size, avg green size, std green size, avg red density, std red density,\
    avg green density, std green density, avg red median size, std red median size, avg green median size, std green median size,\
    avg red pts size, std red pts size, avg green pts size, std green pts size\n")

    # Filtered files:
    if len(green_filess) > 0:
        for green_name in green_filess:
            for red_name in red_filess:
                g = green_name.find('green.csv')
                r = red_name.find('red.csv')
                green_str = green_name[:g]
                red_str = red_name[:r]
                if green_str == red_str:
                    cntr += 1
                    print(cntr)
                    file_directory = main_folder + "/analysis{}".format(get_name(green_name, cntr)) + "/"
                    print(file_directory)
                    green_file_name = green_name
                    red_file_name = red_name
                    proj_name = "test_{}".format(get_name(green_name, cntr))

                    # Execute main function
                    return_list = main(data_type, epsilon, minimum_neighbors, green_file_name, red_file_name, proj_name,
                                       file_directory)
                    # Separate return list to Red and Green
                    red_list = return_list[0]
                    green_list = return_list[1]
                    # Write to file
                    avgd_line = get_res(red_list, green_list, cntr)
                    print(avgd_line)
                    session_data.write(avgd_line)
                    print("END A FILE")  # end of one execution

    # Raw files
    if len(raw_green_filess) > 0:
        for green_name in raw_green_filess:
            for red_name in raw_red_filess:
                g = max(green_name.find("green_raw.csv"), green_name.find("green_row.csv"))
                r = max(red_name.find("red_raw.csv"), red_name.find("red_row.csv"))
                green_str = green_name[:g]
                red_str = red_name[:r]
                if green_str == red_str:
                    cntr += 1
                    print(cntr)
                    file_directory = main_folder + "/raw_analysis{}".format(get_name(green_name, cntr)) + "/"
                    print(file_directory)
                    green_file_name = green_name
                    red_file_name = red_name
                    proj_name = "test_{}".format(get_name(green_name, cntr))

                    return_list = main(data_type, epsilon, minimum_neighbors, green_file_name, red_file_name, proj_name,
                                       file_directory)
                    red_list = return_list[0]
                    green_list = return_list[1]
                    avgd_line = get_res(red_list, green_list, cntr)
                    print(avgd_line)
                    session_data.write(avgd_line)
                    print("END A FILE")

    session_data.close()
    print("END")


# ("color, #points, #red points, #green points, sphere score, angle_x, angle_y, size, density\n")
def get_res(red_list, green_list, cntr):
    len_r = len(red_list)
    len_g = len(green_list)

    for i in range(len_r):
        red_list[i] = red_list[i].split(",")[
                      1:]  # now we got:#points,  #red points, #green points, sphere score, angle_x, angle_y, size, density, median size
        for j in range(len(red_list[i])):
            red_list[i][j] = float(red_list[i][j])
    for i in range(len_g):
        green_list[i] = green_list[i].split(",")[
                        1:]  # now we got:#points,  #red points, #green points, sphere score, angle_x, angle_y, size, density, median size
        for j in range(len(green_list[i])):
            green_list[i][j] = float(green_list[i][j])

    # RED CLUSTERS
    g_in_r_list = []
    for a_list in red_list:
        g_in_r_list.append(a_list[2] / (a_list[1] + a_list[2]))
    if len(g_in_r_list) < 2:
        avgd_line = "{},N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,\
                N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a\n".format(cntr)
        return avgd_line

    avg_per_green_in_red = np.mean(g_in_r_list)
    std_per_green_in_red = statistics.stdev(g_in_r_list)

    # Easier handling numpy arrays
    red_array = np.array(red_list)
    red_means = np.mean(red_array, axis=0)
    red_stds = np.std(red_array, ddof=1, axis=0)
    # Sphere score
    red_average_sphere_score = red_means[3]
    red_std_sphere_score = red_stds[3]
    # X angle
    red_average_angle_x = red_means[4]
    red_std_angle_x = red_stds[4]
    # Y angle
    red_average_angle_y = red_means[5]
    red_std_angle_y = red_stds[5]
    # Size
    red_average_size = red_means[6]
    red_std_size = red_stds[6]
    # Density
    red_avg_naive_density = red_means[7]
    red_std_naive_density = red_stds[7]
    # median Size
    red_average_med_size = red_means[8]
    red_std_med_size = red_stds[8]
    # size  in points
    red_avg_size_pts = red_means[0]
    red_std_size_pts = red_stds[0]

    # GREEN CLUSTERS
    r_in_g_list = []
    for a_list in green_list:
        r_in_g_list.append(a_list[1] / (a_list[1] + a_list[2]))
    if len(r_in_g_list) < 2:
        avgd_line = "{},N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,\
                N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a,N\a\n".format(cntr)
        return avgd_line
    avg_per_red_in_green = np.mean(r_in_g_list)
    std_per_red_in_green = statistics.stdev(r_in_g_list)

    # Transform to numpy
    green_array = np.array(green_list)
    green_means = np.mean(green_array, axis=0)  # calculates the avg of all the columns of the array
    green_stds = np.std(green_array, ddof=1,
                        axis=0)  # calculates the std (sample; ddof =1) of all the columns of the array
    # Sphere score
    green_average_sphere_score = green_means[3]
    green_std_sphere_score = green_stds[3]
    # X angle
    green_average_angle_x = green_means[4]
    green_std_angle_x = green_stds[4]
    # Y angle
    green_average_angle_y = green_means[5]
    green_std_angle_y = green_stds[5]
    # Size
    green_average_size = green_means[6]
    green_std_size = green_stds[6]
    # Density
    green_avg_naive_density = green_means[7]
    green_std_naive_density = green_stds[7]
    # median Size
    green_average_med_size = green_means[8]
    green_std_med_size = green_stds[8]
    # size  in points
    green_avg_size_pts = green_means[0]
    green_std_size_pts = green_stds[0]

    # Sample density
    sample_size = float(2000 * 2000)  # this is for 2d
    clstrs_tot = 0
    # for clst_size in green_array.T[:,5]:
    #     clstrs_tot += clst_size*(math.pi)
    # for clst_size in red_array.T[:,5]:
    #     clstrs_tot += clst_size*(math.pi)
    smpl_density = clstrs_tot / sample_size  # need to add this to the summary file
    # create the line to be written to .csv file
    avgd_line = "{},{},{},{},{},{},{},{},{},{},{},{},{},\
                {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(cntr \
                                                                                      , avg_per_green_in_red \
                                                                                      , std_per_green_in_red \
                                                                                      , avg_per_red_in_green \
                                                                                      , std_per_red_in_green \
                                                                                      , red_average_sphere_score \
                                                                                      , red_std_sphere_score \
                                                                                      , green_average_sphere_score \
                                                                                      , green_std_sphere_score \
                                                                                      , red_average_angle_x \
                                                                                      , red_std_angle_x \
                                                                                      , red_average_angle_y \
                                                                                      , red_std_angle_y \
                                                                                      , green_average_angle_x \
                                                                                      , green_std_angle_x \
                                                                                      , green_average_angle_y \
                                                                                      , green_std_angle_y \
                                                                                      , red_average_size \
                                                                                      , red_std_size \
                                                                                      , green_average_size \
                                                                                      , green_std_size \
                                                                                      , red_avg_naive_density \
                                                                                      , red_std_naive_density \
                                                                                      , green_avg_naive_density \
                                                                                      , green_std_naive_density
                                                                                      , red_average_med_size \
                                                                                      , red_std_med_size \
                                                                                      , green_average_med_size \
                                                                                      , green_std_med_size \
                                                                                      , red_avg_size_pts \
                                                                                      , red_std_size_pts \
                                                                                      , green_avg_size_pts \
                                                                                      , green_std_size_pts)

    return avgd_line


def get_name(file_name, cntr):
    """
    :param file_name: the path of the file containing the name of the test
    :param cntr:  default return if not successful
    :return: the name of the test clean of unrelevant info and file path
    """
    pre = file_name.split('/')[-1]
    pre = pre.split('\\')[-1]
    name = str(cntr)
    if pre is not None:
        ind_bott = pre.rfind('_')
        pre = pre[:ind_bott] if ind_bott != -1 else pre
        if pre.find("green") != -1:
            ind_bott = pre.rfind('_')
            pre = pre[:ind_bott] if ind_bott != -1 else pre
        name = pre
    return name


def get_name_super(folder_path):
    pre = folder_path.split('/')[-1]
    return pre
