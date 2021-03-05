
"""
****------------------------------------SHOW-------------------------------****
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

X_LIM = [0,20000]
Y_LIM = X_LIM
def plot_knn(points, file_directory, name, x = None, y = [0,850], show=False):
    plt.clf()

    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(points)

    distances, indices = nbrs.kneighbors(points)

    t0 = [p[0] for p in distances]
    t0.sort()
    t2 = [p[2] for p in distances]
    t2.sort()
    t4 = [p[4] for p in distances]
    t4.sort()
    t6 = [p[6] for p in distances]
    t6.sort()
    t8 = [p[8] for p in distances]
    t8.sort()
    t10 = [p[10] for p in distances]
    t10.sort()
    t12 = [p[12] for p in distances]
    t12.sort()
    t16 = [p[16] for p in distances]
    t16.sort()

    plt.plot( t16, 'g^', lw = 2, ms = 5)
# bla bla
    if x != None:
        plt.xlim(x)

    if y != None:
        plt.ylim(y)

    plt.ylabel('Distance (nm)', fontsize=16)
    plt.xlabel('Number of Localizations')

    if show==True:
        plt.show()
    plt.savefig(file_directory+name+".png")
    plt.clf()

def get_cluster_picture(s, factor=50, name="", for_all=False):
    plt.clf()
    out_file_name = s.path + "/" + name + ".png"

    colors = []
    points = []
    if not for_all:
        for cluster in s.red_clusters:
            for point in cluster.points:
                colors.append((0, 1, 0) if point.color == "green" else (1, 0, 0))
                points.append(point)
        for cluster in s.green_clusters:
            for point in cluster.points:
                colors.append((0, 1, 0) if point.color == "green" else (1, 0, 0))
                points.append(point)
    else:
        for cluster in s.clusters_red:
            for point in cluster.points:
                colors.append((0, 1, 0) if point.color == "green" else (1, 0, 0))
                points.append(point)
        for cluster in s.clusters_green:
            for point in cluster.points:
                colors.append((0, 1, 0) if point.color == "green" else (1, 0, 0))
                points.append(point)

    x_all = np.array([])
    y_all = np.array([])
    c_all = np.array([])
    s_all = np.array([])

    x = np.array([x.point[0] for x in points],\
    dtype = np.float)
    y = np.array([y.point[1] for y in points],\
    dtype = np.float)
    c = np.array(colors, dtype=np.float)
    s = np.array([50 for k in range(len(x))], dtype=np.float)
    x_all = np.concatenate((x_all, x), axis=0)
    y_all = np.concatenate((y_all, y), axis=0)
    c_all = c
    s_all = np.concatenate((s_all, s), axis=0)
# plt.scatter(x_unclstrd, y_unclstrd, c=c_unclstrd, s=s_unclstrd)

    plt.figure(figsize=(30,30))
    plt.tick_params(axis='both', which='major', labelsize=55)

    plt.scatter(x_all, y_all, c=c_all, s=s_all, alpha=0.5)
    plt.xlim(X_LIM)
    plt.ylim(Y_LIM)

    plt.savefig(out_file_name)

    plt.clf()
    plt.close()


def get_points_picture(s, factor=50, name="", for_all=False):
    plt.clf()
    out_file_name = s.path + "/" + name + ".png"

    colors = []
    points = []
    for point in s.points:
        if point.cluster != -1:
            if point.color == "green":
                colors.append((0, 1, 0))
            else:
                colors.append((1, 0, 0))
        else:
            colors.append((0, 0, 1))
        points.append(point)

    x_all = np.array([])
    y_all = np.array([])
    s_all = np.array([])

    x = np.array([x.point[0] for x in points],\
    dtype = np.float)
    y = np.array([y.point[1] for y in points],\
    dtype = np.float)
    c = np.array(colors, dtype=np.float)
    s = np.array([50 for k in range(len(x))], dtype=np.float)
    x_all = np.concatenate((x_all, x), axis=0)
    y_all = np.concatenate((y_all, y), axis=0)
    c_all = c
    s_all = np.concatenate((s_all, s), axis=0)
# plt.scatter(x_unclstrd, y_unclstrd, c=c_unclstrd, s=s_unclstrd)

    plt.figure(figsize=(30,30))
    plt.tick_params(axis='both', which='major', labelsize=55)

    plt.scatter(x_all, y_all, c=c_all, s=s_all, alpha=0.5)
    plt.xlim(X_LIM)
    plt.ylim(Y_LIM)

    plt.savefig(out_file_name)

    plt.clf()
    plt.close()

def get_picture(s, list_of_points=[], list_of_features=[0], factor=50, name=""):
    plt.clf()
    if name == "":
        name = s.name
        if name == "":
            name = "all"

    out_file_name = s.path + "/" + name + ".png"

    if list_of_points == []:

        x_unclstrd = np.array([x.point[0] for x in s.unclustered_points], dtype = np.float)
        y_unclstrd = np.array([y.point[1] for y in s.unclustered_points], dtype = np.float)
        c_unclstrd = np.array([0 for i in range(len(x_unclstrd))], dtype=np.float)
        s_unclstrd = np.array([50 for i in range(len(x_unclstrd))], dtype=np.float)

        x_g = np.array([x.point[0] for x in s.green_points if x.green_cluster != -1], dtype = np.float)
        y_g = np.array([y.point[1] for y in s.green_points if y.green_cluster != -1], dtype = np.float)
        c_g = np.array([0.5 for i in range(len(x_g))], dtype=np.float)
        s_g = np.array([50 for i in range(len(x_g))], dtype=np.float)

        x_r = np.array([x.point[0] for x in s.red_points if x.red_cluster != -1], dtype = np.float)
        y_r = np.array([y.point[1] for y in s.red_points if y.red_cluster != -1], dtype = np.float)
        c_r = np.array([0.99 for i in range(len(x_r))], dtype=np.float)
        s_r = np.array([50 for i in range(len(x_r))], dtype=np.float)

        x_all = np.concatenate((x_unclstrd, x_g, x_r), axis=0)
        y_all = np.concatenate((y_unclstrd, y_g, y_r), axis=0)
        c_all = np.concatenate((c_unclstrd, c_g, c_r), axis=0)
        s_all = np.concatenate((s_unclstrd, s_g, s_r), axis=0)

    else:
        m = len(list_of_features)

        colors = [0 for i in range(m)]
        for i in range(m):
            if list_of_features[i] == "green":
                colors[i] = 0.5
            elif list_of_features[i] == "red":
                colors[i] = 0.99

        x_all = np.array([])
        y_all = np.array([])
        c_all = np.array([])
        s_all = np.array([])

        for i in range(m):
            x = np.array([x.point[0] for x in list_of_points[i]],\
            dtype = np.float)
            y = np.array([y.point[1] for y in list_of_points[i]],\
            dtype = np.float)
            c = np.array([colors[i] for k in range(len(x))], dtype=np.float)
            s = np.array([50 for k in range(len(x))], dtype=np.float)
            x_all = np.concatenate((x_all, x), axis=0)
            y_all = np.concatenate((y_all, y), axis=0)
            c_all = np.concatenate((c_all, c), axis=0)
            s_all = np.concatenate((s_all, s), axis=0)
# plt.scatter(x_unclstrd, y_unclstrd, c=c_unclstrd, s=s_unclstrd)

    plt.figure(figsize=(30,30))

    plt.scatter(x_all, y_all, c=c_all, s=s_all, alpha=0.5)
    plt.legend()
    rc_orig_params = plt.rcParams.copy()
    # plt.rcParams.update({'font.size': 48})
    plt.tick_params(axis='both', which='major', labelsize=55)
    plt.xlim(X_LIM)
    plt.ylim(Y_LIM)

    plt.savefig(out_file_name)
    # plt.rcParams = rc_orig_params
    # plt.rcParams = plt.rcParams.default

    plt.clf()
    plt.close()

def rainbow(s, other="", for_all = False):
    plt.clf()
    out_file_name = s.path + "/" +"rainbow" + other + ".png"
    color = float(0)
    x_allr = np.array([])
    y_allr = np.array([])
    c_allr = np.array([])
    s_allr = np.array([])
    x_allg = np.array([])
    y_allg = np.array([])
    c_allg = np.array([])
    s_allg = np.array([])

    plt.figure(figsize=(30,30))
    if for_all == False:
        for i in range(len(s.red_clusters)):
            l = len(np.array([x.point[0]\
            for x in s.red_clusters[i].points], dtype = np.float))
            x_allr = np.concatenate((x_allr, np.array([x.point[0]\
            for x in s.red_clusters[i].points], dtype = np.float)), axis=0)
            y_allr = np.concatenate((y_allr, np.array([y.point[1]\
            for y in s.red_clusters[i].points], dtype = np.float)), axis=0)
            c_allr = np.concatenate((c_allr, np.array([color for k in range(l)], dtype=np.float)), axis=0)
            s_allr= np.concatenate((s_allr, np.array([90 for k in range(l)], dtype=np.float)), axis=0)
            color = (color+0.07)%1

        plt.scatter(x_allr, y_allr, c=c_allr, marker='^', s=s_allr, alpha=0.5)

        for i in range(len(s.green_clusters)):

            l = len(np.array([x.point[0]\
            for x in s.green_clusters[i].points], dtype = np.float))
            x_allg = np.concatenate((x_allg, np.array([x.point[0]\
            for x in s.green_clusters[i].points], dtype = np.float)), axis=0)

            y_allg = np.concatenate((y_allg, np.array([y.point[1]\
            for y in s.green_clusters[i].points], dtype = np.float)), axis=0)

            c_allg = np.concatenate((c_allg, np.array([color for k in \
            range(l)], dtype=np.float)), axis=0)

            s_allg = np.concatenate((s_allg, np.array([90 for k in \
            range(l)], dtype=np.float)), axis=0)

            color = (color+0.17)%1

        plt.scatter(x_allg, y_allg, c=c_allg, s=s_allg, alpha=0.5)
    else:
        for i in range(len(s.clusters_red)):
            l = len(np.array([x.point[0]\
            for x in s.clusters_red[i].points], dtype = np.float))
            x_allr = np.concatenate((x_allr, np.array([x.point[0]\
            for x in s.clusters_red[i].points], dtype = np.float)), axis=0)
            y_allr = np.concatenate((y_allr, np.array([y.point[1]\
            for y in s.clusters_red[i].points], dtype = np.float)), axis=0)
            c_allr = np.concatenate((c_allr, np.array([color for k in range(l)], dtype=np.float)), axis=0)
            s_allr= np.concatenate((s_allr, np.array([90 for k in range(l)], dtype=np.float)), axis=0)
            color = (color+0.07)%1

        plt.scatter(x_allr, y_allr, c=c_allr, marker='^', s=s_allr, alpha=0.5)

        for i in range(len(s.clusters_green)):

            l = len(np.array([x.point[0]\
            for x in s.clusters_green[i].points], dtype = np.float))
            x_allg = np.concatenate((x_allg, np.array([x.point[0]\
            for x in s.clusters_green[i].points], dtype = np.float)), axis=0)

            y_allg = np.concatenate((y_allg, np.array([y.point[1]\
            for y in s.clusters_green[i].points], dtype = np.float)), axis=0)

            c_allg = np.concatenate((c_allg, np.array([color for k in \
            range(l)], dtype=np.float)), axis=0)

            s_allg = np.concatenate((s_allg, np.array([90 for k in \
            range(l)], dtype=np.float)), axis=0)

            color = (color+0.17)%1

        plt.scatter(x_allg, y_allg, c=c_allg, s=s_allg, alpha=0.5)

    plt.legend()
    # rc_orig_params = plt.rcParams.copy()
    # plt.rcParams.update({'font.size': 48})
    plt.tick_params(axis='both', which='major', labelsize=55)
    plt.xlim(X_LIM)
    plt.savefig(out_file_name)
    # plt.rcParams = plt.rcParams.default

    # plt.rcParams = rc_orig_params
    plt.clf()
    plt.close()


def show_picture(list_of_points, list_of_features=[], factor=50, dim=2):
    plt.clf()
    xp = np.array([x.point[0] for x in list_of_points], dtype=np.float)
    yp = np.array([x.point[1] for x in list_of_points], dtype=np.float)
#    zp = np.array([x.point[2] for x in list_of_points], dtype=np.float)
    cp = np.array([1 for x in range(len(xp))])
    sp = np.array([90 for x in range(len(xp))])

#    if list_of_features != []:
    xf = np.array([x[0] for x in list_of_features], dtype=np.float)
    yf = np.array([x[1] for x in list_of_features], dtype=np.float)
#    zf = np.array([x[2] for x in list_of_features], dtype=np.float)
    cf = np.array([0.5 for x in range(len(xf))])
    sf = np.array([150 for x in range(len(xf))])
    #plt.scatter(x_unclstrd, y_unclstrd, c=c_unclstrd, s=s_unclstrd)

    x_all = np.concatenate(xp, xf)
    y_all = np.concatenate(yp, yf)
#    z_all = np.concatenate(zp, zf)
    c_all = np.concatenate(cp, cf)
    s_all = np.concatenate(sp, sf)
    plt.figure(figsize=(50,50))
    plt.scatter(x_all, y_all, c=c_all, s=s_all, alpha=0.5)

    plt.show()

    plt.clf()
    plt.close()

def make_histogram(s, list, color, other=""):
    out_file_name = s.path + "/" +"hist_" + color + other + ".png"
    if len(list) < 2: # no plotting in this case
        plt.title("less than 2 clusters")
    else:
        # plt.hist(list, normed=True)
        weights = np.ones_like(list)/float(len(list))
        plt.hist(list, weights = weights, color='slategrey')
        # plt.hist(list, bins=[0,60,120,180, 240, 300, 360, 420, 480, 1000], weights=weights)
        plt.ylabel('Percentage of Clusters')
        plt.xlabel('Mean Diameter (nm)')
    plt.savefig(out_file_name)
    plt.clf()
    plt.close()
