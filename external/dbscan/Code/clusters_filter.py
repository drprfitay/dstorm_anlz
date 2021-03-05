__author__ = 'UriA12'
import os
import re
import time
import csv
from parse_super_detect import get_res

DEBUG = False


def filter_it(color_a, points, red_points, green_points, density, coloc_a, anglex, angley, size, files_path, dest_path,
              source="no",
              name=""):

    out_file = open(os.path.normcase(os.path.join(dest_path, "filtered_summary_{}_{}.csv".format(name, time.strftime(
        "%Y-%m-%d_%H-%M-%S", time.gmtime())))), "w")

    csv_titles = "test#,total_number_of_points,total_number_of_red_points,total_number_of_green_points,total_number_of_clustered_points,\
            total_number_of_unclustered_points,total_number_of_clustered_red_points,total_number_of_clustered_green_points,\
            relative_clustered_points,relative_unclustered_points,relative_red_clustered_points,relative_green_clustered_points,\
            #clusters,#red clusters,#green clusters,avg green in red clusters,std green in red clusters,avg red in green clusters,\
            std red in green clusters,avg red sphericity,std red sphericity,avg green sphericity,std green sphericity,\
            avg red Xangl,std red Xangl,avg red Yangl,std red Yangl,avg green Xangl,std green Xangl,avg green Yangl,\
            std green Yangl,avg red size,std red size,avg green size,std green size,sample density,avg red density,std red density,\
            avg green density,std green density,red median size,green median size,\
            avg red pts size,std red pts size,avg green pts size,std green pts size,colocalization %green in red,colocalization %red in green,test name\n"
    out_file.write(csv_titles)

    if source != "no":
        # get the source details
        tot_points = []
        tot_red = []
        tot_green = []
        tot_clustered = []
        tot_clustered_red = []
        tot_clustered_green = []
        unclstrd_tot = []
        unclstrd_green = []
        unclstrd_red = []
        test_names = []
        with open(source, "r") as f:
            f = csv.reader(f, delimiter=',')
            next(f)
            for row in f:
                tot_points.append(int(row[1]))
                tot_red.append(int(row[2]))
                tot_green.append(int(row[3]))
                tot_clustered.append(int(row[4]))
                tot_clustered_red.append(int(row[6]))
                tot_clustered_green.append(int(row[7]))
                unclstrd_tot.append(int(row[5]))
                unclstrd_red.append((int(row[2]) - int(row[6])))
                unclstrd_green.append((int(row[3]) - int(row[7])))
                test_names.append(row[48])

    MAX = 1000000.
    MIN = -1.

    color = color_a
    coloc = coloc_a
    if DEBUG: print(points)
    points_lst = points.split(";")
    if DEBUG: print(points_lst)

    points_min = MIN if points_lst[0] == 'MIN' else float(points_lst[0])
    points_max = MAX if points_lst[1] == "MAX" else float(points_lst[1])
    if DEBUG: print(red_points)

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

    anglex_lst = anglex.split(";")
    anglex_min = MIN if anglex_lst[0] == "MIN" else float(anglex_lst[0])
    anglex_max = MAX if anglex_lst[1] == "MAX" else float(anglex_lst[1])

    angley_lst = angley.split(";")
    angley_min = MIN if angley_lst[0] == "MIN" else float(angley_lst[0])
    angley_max = MAX if angley_lst[1] == "MAX" else float(angley_lst[1])

    size_lst = size.split(";")
    size_min = MIN if size_lst[0] == "MIN" else float(size_lst[0])
    size_max = MAX if size_lst[1] == "MAX" else float(size_lst[1])


    # open clusters file
    clusters_file = open(os.path.normcase(os.path.join(dest_path, "filtered_clusters_{}_{}.csv".format(name,
                                                                                                       time.strftime(
                                                                                                           "%Y-%m-%d_%H-%M-%S",
                                                                                                           time.gmtime())))),
                         "w")
    csv_clusters_titles = "color,#points,#red points,#green points,sphere score,angle_x,angle_y,size,density,x,y,z,colocalized,from file\n"
    clusters_file.write(csv_clusters_titles)

    for a_file in files_path:
        if not re.findall(r".*?\.csv", a_file, re.DOTALL):
            return 1
    if dest_path == "": return 2
    cnt = -1
    for a_file in files_path:
        cnt += 1
        red_list_dicts = []
        green_list_dicts = []
        red_list = []
        green_list = []
        points_counter = [0, 0, 0]
        with open(a_file) as csvfile:
            if DEBUG: print("opened")
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row == []: continue
                if DEBUG: print(row)
                if color != "both":
                    if row['color'] != color:
                        points_counter[0] += int(row['#points'])
                        points_counter[1] += int(row['#red points'])
                        points_counter[2] += int(row['#green points'])
                        continue
                if float(row['#points']) > points_max or float(row['#points']) < points_min:
                    points_counter[0] += int(row['#points'])
                    points_counter[1] += int(row['#red points'])
                    points_counter[2] += int(row['#green points'])
                    continue
                if float(row['#red points']) > red_points_max or float(row['#red points']) < red_points_min:
                    points_counter[0] += int(row['#points'])
                    points_counter[1] += int(row['#red points'])
                    points_counter[2] += int(row['#green points'])
                    continue
                if float(row['#green points']) > green_points_max or float(row['#green points']) < green_points_min:
                    points_counter[0] += int(row['#points'])
                    points_counter[1] += int(row['#red points'])
                    points_counter[2] += int(row['#green points'])
                    continue
                if float(row['density']) > density_max or float(row['density']) < density_min:
                    points_counter[0] += int(row['#points'])
                    points_counter[1] += int(row['#red points'])
                    points_counter[2] += int(row['#green points'])
                    continue
                if coloc != "all":
                    if row['colocalized'] == '0' and coloc == 'yes':
                        points_counter[0] += int(row['#points'])
                        points_counter[1] += int(row['#red points'])
                        points_counter[2] += int(row['#green points'])
                        continue
                    if row['colocalized'] == '1' and coloc == 'no':
                        points_counter[0] += int(row['#points'])
                        points_counter[1] += int(row['#red points'])
                        points_counter[2] += int(row['#green points'])
                        continue

                if float(row['angle_x']) > anglex_max or float(row['angle_x']) < anglex_min:
                        points_counter[0] += int(row['#points'])
                        points_counter[1] += int(row['#red points'])
                        points_counter[2] += int(row['#green points'])
                        continue
                if float(row['angle_y']) > angley_max or float(row['angle_y']) < angley_min:
                        points_counter[0] += int(row['#points'])
                        points_counter[1] += int(row['#red points'])
                        points_counter[2] += int(row['#green points'])
                        continue

                if float(row['size']) > size_max or float(row['size']) < size_min:
                    points_counter[0] += int(row['#points'])
                    points_counter[1] += int(row['#red points'])
                    points_counter[2] += int(row['#green points'])
                    continue
                if color == "green" and row['color'] == "green":
                    green_list_dicts.append(row)
                else:
                    red_list_dicts.append(row)
                row_x = 'x' + ","
                row_y = 'y' + ","
                row_z = 'z' + ","
                try:
                    row_x = row['x'] + ","
                    row_y = row['y'] + ","
                    row_z = row['z'] + ","
                    #  break
                except KeyError:
                    pass
                if DEBUG: print("writing to file")
                # write the cluster to the clusters file
                str_row = row['color'] + "," + \
                          row['#points'] + "," + \
                          row['#red points'] + "," + \
                          row['#green points'] + "," + \
                          row['sphere score'] + "," + \
                          row['angle_x'] + "," + \
                          row['angle_y'] + "," + \
                          row['size'] + "," + \
                          row['density'] + "," + \
                          row_x + \
                          row_y + \
                          row_z + \
                          row['colocalized'] + "," + \
                          get_name(a_file) + "\n"
                if DEBUG: print(str_row)
                clusters_file.write(str_row)

        for dic in red_list_dicts:
            this_color = dic['color']
            line = dic['color'] + "," + \
                   dic['#points'] + "," + \
                   dic['#red points'] + "," + \
                   dic['#green points'] + "," + \
                   dic['sphere score'] + "," + \
                   dic['angle_x'] + "," + \
                   dic['angle_y'] + "," + \
                   dic['size'] + "," + \
                   dic['density'] + "," + \
                   dic['x'] + "," + \
                   dic['y'] + "," + \
                   dic['z'] + "," + \
                   dic['colocalized'] + "\n"
            if this_color == "green":
                green_list.append(line)
            else:
                red_list.append(line)

        b_list = ["NaN" for i in range(12)]
        if source != "no!":
            # basics stuff
            b_list[0] = tot_points[cnt]
            b_list[1] = tot_red[cnt]
            b_list[2] = tot_green[cnt]
            b_list[3] = tot_clustered[cnt] - (unclstrd_tot[cnt] + points_counter[0])
            b_list[4] = unclstrd_tot[cnt] + points_counter[0]
            b_list[5] = tot_clustered_red[cnt] - points_counter[1]
            b_list[6] = tot_clustered_green[cnt] - points_counter[2]
            b_list[7] = float(b_list[3])/float(tot_points[cnt])
            b_list[8] = float(b_list[4])/float(tot_points[cnt])
            b_list[9] = float(b_list[5])/float(tot_red[cnt])
            b_list[10] = float(b_list[6])/float(tot_green[cnt])

        filters = "points@{}:{}::red_points@{}:{}::green_points@{}:{}::density@{}:{}::size@{}:{}::color@{}::colocalization@{}::".format(
            points_min, \
            points_max, \
            red_points_min, \
            red_points_max, \
            green_points_min, \
            green_points_max, \
            density_min, \
            density_max, \
            size_min, \
            size_max, \
            color, \
            coloc)

        avgd_line = get_res(red_list, green_list, cnt, filters, b_list)
        out_file.write(avgd_line)

    out_file.close()
    # close the clusters file
    clusters_file.close()
    return 0

    if DEBUG: print("done")


def get_name(file_name):
    """
    :param file_name: the path of the file containing the name of the test
    :param cntr:  default return if not successful
    :return: the name of the test clean of unrelevant info and file path
    """
    pre = file_name.split('/')[-2]
    name = str(0)
    if pre is not None:
        name = pre
    return name
