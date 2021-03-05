__author__ = 'UriA12'
import os
import re
import time
import statistics
import math
import numpy as np
from parse_main import main
import warnings
from unclustered import main as unclustered
from unclustered import get_unclustered_by_filter_main

debug = False
np.seterr(all='ignore')
warnings.filterwarnings("ignore")


# main function, get info from "main_with_gui"

def go(eps, min_ngbs, mini_eps, mini_min_ngbs, d_type, pth, mode, f_color, f_points, f_red_points,
       f_green_points, f_density, f_coloc, f_x_angle, f_y_angle, f_size):
    print(eps, min_ngbs, mini_eps, mini_min_ngbs, d_type, pth)
    data_type = d_type  # "2d", etc.
    epsilon = eps
    minimum_neighbors = min_ngbs
    mini_epsilon = mini_eps
    mini_minimum_neighbors = mini_min_ngbs

    main_folder = pth  # where all the sub_folders are at #should be session folder
    directories = []
    for root, dirs, files in os.walk(main_folder, topdown=True):
        for dir in dirs:
            directories.append(os.path.join(root, dir))
    directories.append(main_folder)
    print(directories)
    for directory in directories:
        print(directory)
        print(os.listdir(directory))
        filessss = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        print(filessss)
        files = [os.path.join(directory, f) for f in filessss]
        if len(files) == 0:
            continue
        session_name = get_name_super(directory)
        particles_filess = []
        cntr = 0
        green_filess = []
        red_filess = []
        raw_green_filess = []
        raw_red_filess = []
        old_filess = []

        nrm_cntr = 0
        raw_cntr = 0
        prtcls_cntr = 0
        old_cntr = 0
        for name in files:
            if re.findall(r".*?done\.txt", name, re.DOTALL):
                print("Done file found!")
                break
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
                # if re.findall(r".*?regions.*?\.txt", name, re.DOTALL):
                #     old_filess.append(os.path.join(root, name))
                # # if re.findall(r"particles\.csv", name):
                #     if not os.path.join(root, name) in final_particles_files: # NOTICE!!! NOT SUPPORTED YET!
                #         particles_filess.append(os.path.join(root, name))
                #         prtcls_cntr += 1
                #         print("particles counter: {}".format(prtcls_cntr))

        if (len(green_filess) + len(raw_green_filess)) or (len(red_filess) + len(raw_red_filess)) > 0:
            new_directory = directory + "/test_{}_eps{}_min{}_{}".format(
                time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()), epsilon, minimum_neighbors, d_type)
            # make output folder and open files
            if not os.path.exists(new_directory):
                os.mkdir(new_directory)

            session_data = open(os.path.normcase(os.path.join(new_directory, session_name + "_final_summary.csv")), "w")
            session_data_pre = open(os.path.normcase(os.path.join(new_directory, session_name + "_pre_summary.csv")),
                                    "w")
            session_data_all = open(os.path.normcase(os.path.join(new_directory, session_name + "_all_summary.csv")),
                                    "w")
            done_file = open(os.path.normcase(os.path.join(new_directory, "done.txt")), "w")
            csv_titles = "test#,total_number_of_points,total_number_of_red_points,total_number_of_green_points,total_number_of_clustered_points,\
            total_number_of_unclustered_points,total_number_of_clustered_red_points,total_number_of_clustered_green_points,\
            relative_clustered_points,relative_unclustered_points,relative_red_clustered_points,relative_green_clustered_points,\
            #clusters,#red clusters,#green clusters,avg green in red clusters,std green in red clusters,avg red in green clusters,\
            std red in green clusters,avg red sphericity,std red sphericity,avg green sphericity,std green sphericity,\
            avg red Xangl,std red Xangl,avg red Yangl,std red Yangl,avg green Xangl,std green Xangl,avg green Yangl,\
            std green Yangl,avg red size,std red size,avg green size,std green size,sample density,avg red density,std red density,\
            avg green density,std green density,red median size,green median size,\
            avg red pts size,std red pts size,avg green pts size,std green pts size,colocalization %green in red,colocalization %red in green,test name\n"

            session_data.write(csv_titles)
            session_data_pre.write(csv_titles)
            session_data_all.write(csv_titles)
            done_file.write("This folder is done with.\n")
            done_file.close()
        if mode == 1:  # single protein analysis
            print("--Single Protein Analysis--")
            some_files = [(x, "green") for x in green_filess] + [(x, "red") for x in red_filess]
            for some_file in some_files:
                cntr += 1
                print(cntr)
                file_directory = new_directory + "/analysis_{}_{}".format(get_name(some_file[0], cntr),
                                                                          some_file[1]) + "/"
                print(file_directory)
                color = some_file[1]
                dummy_file = "green_dummy.csv" if some_file[1] == "red" else "red_dummy.csv"
                proj_name = "test_{}".format(get_name(some_file[0], cntr))
                # Execute main function
                return_list = main(data_type, epsilon, minimum_neighbors, mini_epsilon, mini_minimum_neighbors,
                                   some_file[0], dummy_file, proj_name, file_directory, mode) if some_file[
                                                                                                     1] == "green" else main(
                    data_type, epsilon, minimum_neighbors, mini_epsilon, mini_minimum_neighbors, dummy_file,
                    some_file[0], proj_name, file_directory, mode)
                # Separate return list to Red and Green
                red_list = return_list[0]
                green_list = return_list[1]
                red_list_pre = return_list[2]
                green_list_pre = return_list[3]
                basics_list = return_list[4]
                basics_list_pre = return_list[5]
                basics_all = return_list[6]
                red_all = return_list[7]
                green_all = return_list[8]
                sample = return_list[9]
                # unclustered_analysis
                unclustered_by_filter_points = get_unclustered_by_filter_main(sample.pre_red_clusters,
                                                                              sample.pre_green_clusters, f_color, f_points,
                                                                              f_red_points,
                                                                              f_green_points, f_density, f_coloc,
                                                                              f_x_angle, f_y_angle, f_size)

                unclustered(sample, data_type, 20, 6, file_directory, mode=1, color=color, dummy_file=dummy_file,
                            unclustered_by_filter_points=unclustered_by_filter_points)
                # Write to file
                avgd_line = get_res(red_list, green_list, cntr, proj_name, basics_list)
                avgd_line_pre = get_res(red_list_pre, green_list_pre, cntr, proj_name, basics_list_pre)
                avgd_line_all = get_res(red_all, green_all, cntr, proj_name, basics_all)
                print(avgd_line)
                session_data.write(avgd_line)
                session_data_pre.write(avgd_line_pre)
                session_data_all.write(avgd_line_all)
                print("END A FILE")  # end of one execution

        # Filtered files:
        if mode == 2 and len(green_filess) > 0:
            for green_name in green_filess:
                for red_name in red_filess:
                    if debug: print(green_name, red_name)
                    g = green_name.find('green.csv')
                    r = red_name.find('red.csv')
                    if debug: print("indexes:\t", g, r)
                    green_str = green_name[:g]
                    red_str = red_name[:r]
                    if green_str == red_str:
                        cntr += 1
                        print(cntr)
                        file_directory = new_directory + "/analysis_{}".format(get_name(green_name, cntr)) + "/"
                        print(file_directory)
                        green_file_name = green_name
                        red_file_name = red_name
                        proj_name = "test_{}".format(get_name(green_name, cntr))

                        # Execute main function
                        return_list = main(data_type, epsilon, minimum_neighbors, mini_epsilon, mini_minimum_neighbors,
                                           green_file_name, red_file_name, proj_name, file_directory)
                        # Separate return list to Red and Green
                        red_list = return_list[0]
                        green_list = return_list[1]
                        red_list_pre = return_list[2]
                        green_list_pre = return_list[3]
                        basics_list = return_list[4]
                        basics_list_pre = return_list[5]
                        basics_all = return_list[6]
                        red_all = return_list[7]
                        green_all = return_list[8]
                        sample = return_list[9]
                        unclustered_by_filter_points = get_unclustered_by_filter_main(sample.pre_red_clusters,
                                                                                      sample.pre_green_clusters, f_color,
                                                                                      f_points,
                                                                                      f_red_points, f_green_points,
                                                                                      f_density, f_coloc, f_x_angle,
                                                                                      f_y_angle, f_size)
                        # unclustered_analysis
                        unclustered(sample, data_type, 20, 6, file_directory, unclustered_by_filter_points=unclustered_by_filter_points)
                        # Write to file
                        avgd_line = get_res(red_list, green_list, cntr, proj_name, basics_list)
                        avgd_line_pre = get_res(red_list_pre, green_list_pre, cntr, proj_name, basics_list_pre)
                        avgd_line_all = get_res(red_all, green_all, cntr, proj_name, basics_all)
                        print(avgd_line)
                        session_data.write(avgd_line)
                        session_data_pre.write(avgd_line_pre)
                        session_data_all.write(avgd_line_all)
                        print("END A FILE")  # end of one execution

        # Raw files
        if mode == 2 and len(raw_green_filess) > 0:
            for green_name in raw_green_filess:
                for red_name in raw_red_filess:
                    g = max(green_name.find("green_raw.csv"), green_name.find("green_row.csv"))
                    r = max(red_name.find("red_raw.csv"), red_name.find("red_row.csv"))
                    green_str = green_name[:g]
                    red_str = red_name[:r]
                    if green_str == red_str:
                        cntr += 1
                        print(cntr)
                        file_directory = new_directory + "/raw_analysis_{}".format(get_name(green_name, cntr)) + "/"
                        print(file_directory)
                        green_file_name = green_name
                        red_file_name = red_name
                        proj_name = "test_{}".format(get_name(green_name, cntr))

                        return_list = main(data_type, epsilon, minimum_neighbors, mini_epsilon, mini_minimum_neighbors,
                                           green_file_name, red_file_name, proj_name, file_directory)
                        red_list = return_list[0]
                        green_list = return_list[1]
                        red_list_pre = return_list[2]
                        green_list_pre = return_list[3]
                        basics_list = return_list[4]
                        basics_list_pre = return_list[5]
                        basics_all = return_list[6]
                        red_all = return_list[7]
                        green_all = return_list[8]
                        avgd_line = get_res(red_list, green_list, cntr, proj_name, basics_list)
                        avgd_line_pre = get_res(red_list_pre, green_list_pre, cntr, proj_name, basics_list_pre)
                        avgd_line_all = get_res(red_all, green_all, cntr, proj_name, basics_all)
                        print(avgd_line)
                        session_data.write(avgd_line)
                        session_data_pre.write(avgd_line_pre)
                        session_data_all.write(avgd_line_all)
                        print("END A FILE")
                        # old files
        if len(old_filess) > 0:
            session_data = open(os.path.normcase(os.path.join(directory, session_name + "_final_summary.csv")), "w")
            session_data_pre = open(os.path.normcase(os.path.join(directory, session_name + "_pre_summary.csv")), "w")
            csv_titles = "test#,total_number_of_points,total_number_of_red_points,total_number_of_green_points,total_number_of_clustered_points,\
            total_number_of_unclustered_points,total_number_of_clustered_red_points,total_number_of_clustered_green_points,\
            relative_clustered_points,relative_unclustered_points,relative_red_clustered_points,relative_green_clustered_points,\
            #clusters,#red clusters,#green clusters,avg green in red clusters,std green in red clusters,avg red in green clusters,\
            std red in green clusters,avg red sphericity,std red sphericity,avg green sphericity,std green sphericity,\
            avg red Xangl,std red Xangl,avg red Yangl,std red Yangl,avg green Xangl,std green Xangl,avg green Yangl,\
            std green Yangl,avg red size,std red size,avg green size,std green size,sample density,avg red density,std red density,\
            avg green density,std green density,red median size,green median size,\
            avg red pts size,std red pts size,avg green pts size,std green pts size,colocalization %green in red,colocalization %red in green,test name\n"

            session_data.write(csv_titles)
            session_data_pre.write(csv_titles)

            for file_name in old_filess:
                cntr += 1
                print(cntr)
                file_directory = directory + "/old_analysis_{}".format(cntr) + "/"
                print(file_directory)
                proj_name = "test_{}".format(cntr)
                return_list = main("old", epsilon, minimum_neighbors, mini_epsilon, mini_minimum_neighbors, file_name,
                                   "", proj_name, file_directory)
                red_list = return_list[0]
                green_list = return_list[1]
                red_list_pre = return_list[2]
                green_list_pre = return_list[3]
                basics_list = return_list[4]
                basics_list_pre = return_list[5]
                avgd_line = get_res(red_list, green_list, cntr, proj_name, basics_list)
                avgd_line_pre = get_res(red_list_pre, green_list_pre, cntr, proj_name, basics_list_pre)
                print(avgd_line)
                session_data.write(avgd_line)
                session_data_pre.write(avgd_line_pre)
                print("END A FILE")
            session_data.close()
            session_data_pre.close()
        print("END")


# ("color, #points, #red points, #green points, sphere score, angle_x, angle_y, size, density\n")
def get_res(red_list, green_list, cntr, proj_name, b_list):
    len_r = len(red_list)
    len_g = len(green_list)
    for i in range(len_r):
        red_list[i] = red_list[i].split(",")[
                      1:]  # now we got:#points,  #red points, #green points, sphere score, angle_x, angle_y, size, density
        for j in range(len(red_list[i])):
            red_list[i][j] = float(red_list[i][j])
    for i in range(len_g):
        green_list[i] = green_list[i].split(",")[
                        1:]  # now we got:#points,  #red points, #green points, sphere score, angle_x, angle_y, size, density
        for j in range(len(green_list[i])):
            green_list[i][j] = float(green_list[i][j])

    # RED CLUSTERS
    g_in_r_list = []
    for a_list in red_list:
        if len(a_list) < 2: continue
        g_in_r_list.append(a_list[2] / (a_list[1] + a_list[2]))
    # if len(g_in_r_list) < 2:
    #     avgd_line = "{},N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N\n".format(cntr)
    #     return avgd_line

    avg_per_green_in_red = np.mean(g_in_r_list) if len(g_in_r_list) > 2 else 0
    std_per_green_in_red = statistics.stdev(g_in_r_list) if len(g_in_r_list) > 2 else 0

    # Easier handling numpy arrays
    red_array = np.array(red_list)
    # print if in DEBUG:
    if debug:
        print(red_array)
    red_means = np.mean(red_array, axis=0)
    red_check = True if red_array.size else False  # True if check is good
    if debug:
        print(red_means)

    red_stds = np.std(red_array, ddof=1, axis=0)
    red_medians = np.median(red_array, axis=0)
    # Sphere score
    red_average_sphere_score = red_means[3] if red_check else 0
    red_std_sphere_score = red_stds[3] if red_check else 0
    # X angle
    red_average_angle_x = red_means[4] if red_check else 0
    red_std_angle_x = red_stds[4] if red_check else 0
    # Y angle
    red_average_angle_y = red_means[5] if red_check else 0
    red_std_angle_y = red_stds[5] if red_check else 0
    # Size
    red_average_size = red_means[6] if red_check else 0
    red_std_size = red_stds[6] if red_check else 0
    # Density
    red_avg_naive_density = red_means[7] if red_check else 0
    red_std_naive_density = red_stds[7] if red_check else 0
    # median Size
    red_average_med_size = red_medians[6] if red_check else 0
    # size  in points
    red_avg_size_pts = red_means[0] if red_check else 0
    red_std_size_pts = red_stds[0] if red_check else 0
    # colocalization percentage
    sum_of_coloc_r = np.sum(red_array[:, 11]) if len(red_array) > 7 else 0
    per_g_in_r_col = sum_of_coloc_r / len(red_array) if len(red_array) > 0 else 0

    # GREEN CLUSTERS
    r_in_g_list = []
    for a_list in green_list:
        if len(a_list) < 2: continue
        r_in_g_list.append(a_list[1] / (a_list[1] + a_list[2]))
    # if len(r_in_g_list) < 2:
    #     avgd_line = "{},N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N\n".format(cntr)
    #     return avgd_line

    avg_per_red_in_green = np.mean(r_in_g_list) if len(r_in_g_list) > 2 else 0
    std_per_red_in_green = statistics.stdev(r_in_g_list) if len(r_in_g_list) > 2 else 0

    # Transform to numpy
    green_array = np.array(green_list)
    green_means = np.mean(green_array, axis=0)  # calculates the avg of all the columns of the array
    green_check = True if green_array.size else False  # True if check is good

    green_stds = np.std(green_array, ddof=1,
                        axis=0)  # calculates the std (sample; ddof =1) of all the columns of the array
    green_medians = np.median(green_array, axis=0)
    # Sphere score
    green_average_sphere_score = green_means[3] if green_check else 0
    green_std_sphere_score = green_stds[3] if green_check else 0
    # X angle
    green_average_angle_x = green_means[4] if green_check else 0
    green_std_angle_x = green_stds[4] if green_check else 0
    # Y angle
    green_average_angle_y = green_means[5] if green_check else 0
    green_std_angle_y = green_stds[5] if green_check else 0
    # Size
    green_average_size = green_means[6] if green_check else 0
    green_std_size = green_stds[6] if green_check else 0
    # Density
    green_avg_naive_density = green_means[7] if green_check else 0
    green_std_naive_density = green_stds[7] if green_check else 0
    # median Size
    green_average_med_size = green_medians[6] if green_check else 0
    # size  in points
    green_avg_size_pts = green_means[0] if green_check else 0
    green_std_size_pts = green_stds[0] if green_check else 0
    sum_of_coloc_g = np.sum(green_array[:, 11]) if len(green_array) > 7 else 0
    per_r_in_g_col = sum_of_coloc_g / len(green_array) if len(green_array) > 0 else 0

    # How many clusters?
    number_red = len(red_list)
    number_green = len(green_list)
    total_clusters = len(red_list) + len(green_list)

    # Sample density
    sample_size = float(2000 * 2000)  # this is for 2d
    clstrs_tot = 0
    if green_array != []:
        for clst_size in green_array[:, 6]:  # the 'size' column of the array
            clstrs_tot += clst_size * (math.pi)
    if red_array != []:
        for clst_size in red_array[:, 6]:  # the 'size' column of the array
            clstrs_tot += clst_size * (math.pi)
    clstrs_tot /= sample_size  # should be the sample density

    # basics stuff
    total_number_of_points = b_list[0]
    total_number_of_red_points = b_list[1]
    total_number_of_green_points = b_list[2]
    total_number_of_clustered_points = b_list[3]
    total_number_of_unclustered_points = b_list[4]
    total_number_of_clustered_red_points = b_list[5]
    total_number_of_clustered_green_points = b_list[6]
    relative_clustered_points = b_list[7]
    relative_unclustered_points = b_list[8]
    relative_red_clustered_points = b_list[9]
    relative_green_clustered_points = b_list[10]

    # create the line to be written to .csv file
    avgd_line = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},\
                {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(cntr \
                                                                                      , total_number_of_points \
                                                                                      , total_number_of_red_points \
                                                                                      , total_number_of_green_points \
                                                                                      , total_number_of_clustered_points \
                                                                                      ,
                                                                                      total_number_of_unclustered_points \
                                                                                      ,
                                                                                      total_number_of_clustered_red_points \
                                                                                      ,
                                                                                      total_number_of_clustered_green_points \
                                                                                      , relative_clustered_points \
                                                                                      , relative_unclustered_points \
                                                                                      , relative_red_clustered_points \
                                                                                      , relative_green_clustered_points \
                                                                                      , total_clusters \
                                                                                      , number_red \
                                                                                      , number_green \
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
                                                                                      , clstrs_tot \
                                                                                      , red_avg_naive_density \
                                                                                      , red_std_naive_density \
                                                                                      , green_avg_naive_density \
                                                                                      , green_std_naive_density \
                                                                                      , red_average_med_size \
                                                                                      , green_average_med_size \
                                                                                      , red_avg_size_pts \
                                                                                      , red_std_size_pts \
                                                                                      , green_avg_size_pts \
                                                                                      , green_std_size_pts \
                                                                                      , per_g_in_r_col \
                                                                                      , per_r_in_g_col \
                                                                                      , proj_name)

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
        name = pre
    return name


def get_name2(file_name, cntr):
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
        if pre.rfind("green") != -1:
            ind_bott = pre.rfind('_')
            pre = pre[:ind_bott] if ind_bott != -1 else pre
        name = pre
    return name


def get_name_super(folder_path):
    pre = folder_path.split('/')[-1]
    pre = pre.split('\\')[-1]
    return pre
