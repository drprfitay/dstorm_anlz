import threading
import os
import sys
import torch
import numpy as np
import pandas as pd
import threading
import plotly.graph_objects as go
import functools
from enum import Enum
from random import randint, shuffle
from sklearn.neighbors import NearestNeighbors
from IPython.display import display, HTML
from plotly.subplots import make_subplots
from external.datasets.dstorm_datasets import DstormDatasetDBSCAN
from matplotlib import cm, colors
from shutil import rmtree
from scipy import spatial, stats
from csv import writer

export_file = "data.csv"
export_folder = "./static/export/"
scan_results_folder = "./static/scan_results/"
prescan_results_folder = "./static/prescan_results/"

#color_palette = ["#" +  "".join([hex(randint(0,15))[2:] for i in range(0,6)]) for i in range(0,5000)]
st = [0, 'r', 155]
clrs = [[[[a, b,c] for c in st if c != a and c != b] for b in st if b != a] for a in st]
clrs = functools.reduce(lambda a,b: a+b, clrs)
clrs = functools.reduce(lambda a,b: a+b, clrs)
clrsx = [[[randint(0,255) if t == 'r' else t for t in c] for r in range(0,500)] for c in clrs]
clrsx = functools.reduce(lambda a,b: a+b, clrsx)
shuffle(clrsx)
make_clr = lambda l: "#" +  "".join([hex(c)[2:] if c > 15 else '0' + hex(c)[2:] for c in l])
color_palette = [make_clr(l) for l in clrsx]

def make_prefix(conf):
  strbool = lambda b: "T" if b else "F"
  prefix_str = "%s_z%s_nr%s%f_ddt%f_zddt%f_pc%fSSS" %\
    ("H" if conf["general"]["use_hdbscan"] else "D",
     strbool(conf["general"]["use_z"]),
     strbool(conf["general"]["noise_reduce"]),
     conf["general"]["stddev_num"],
     conf["general"]["density_drop_threshold"],
     conf["general"]["threed_drop_threshold"],
     conf["general"]["photon_count"])

  hdbscan_prefix = "alg%s_mn%d_ms%d_eps%d_alp%s" %\
    (conf["hdbscan"]["hdbscan_extracting_alg"],
     conf["hdbscan"]["hdbscan_min_npoints"],
     conf["hdbscan"]["hdbscan_min_samples"],
     conf["hdbscan"]["hdbscan_eps"],
     conf["hdbscan"]["hdbscan_alpha"])

  dbscan_prefix = "mn%d_eps%d_ms%d"  %\
   (conf["dbscan"]["min_npoints"],
    conf["dbscan"]["dbscan_eps"],
    conf["dbscan"]["dbscan_min_samples"])

  return prefix_str + (hdbscan_prefix if conf["general"]["use_hdbscan"] else dbscan_prefix) + "EEE"

def format_exception(e):
  exc_type, exc_obj, exc_tb = sys.exc_info()
  fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
  print("\n\n----------Caught exception!: (", e, ")\n")
  print("----------In line: ", exc_tb.tb_lineno)
  print("----------In file: ", fname) 
  print("----------Exception type: ", exc_type)

class scanThread(threading.Thread):
    
    class scanStatus(Enum):
        SCAN_NOT_STARTED = 1
        SCAN_FINISHED_SUCCESSFULLY = 2
        SCAN_FAILED = 3
        SCAN_RUNNING = 4
        SCAN_GENERATING_PLOTS = 5

    themes = { "1" : ("ggplot2", "grey", "black"), "2" : ("plotly_dark","white", "white"), "3": ("none", "grey", "black")}

    default_configuration = {\
    "dbscan" : 
        { "min_npoints" : 0,
          "dbscan_eps" : 70,
          "dbscan_min_samples" : 22},

    "hdbscan" : 
        { "hdbscan_min_npoints" : 50,
          "hdbscan_min_samples" : 1,
          "hdbscan_eps" : -9999,
          "hdbscan_alpha": "1.0",
          "hdbscan_extracting_alg": "leaf"},

    "coloc"  :    
        {"coloc_distance" : 50,
        "coloc_neighbors" : 1,
        "workers" : 2 },

    "k-dist" : 
        { "n_neighbors" : 65, 
          "algorithm" : "ball_tree",
          "k" : 16,
          "is_3D": True},

     "general" :
        { "use_z": True,
          "use_hdbscan": True,
          "noise_reduce": True,
          "stddev_num": 1.5,
          "photon_count" : 1000.0,
          "density_drop_threshold": 0.0,
          "threed_drop_threshold": 0.0}
    }

    default_prescan = {
      "k": 16,
      "eps": 70
    }


    def __init__(self, is_prescan=False):
        self.ex_files = []
        self.conf = None
        self.status = scanThread.scanStatus.SCAN_NOT_STARTED;
        self.original_df = None
        self.clusters_df = None
        self.global_df = pd.DataFrame([])
        self.global_clusters_df = pd.DataFrame([])
        self.result_json = {}
        self.temp_result_holder = None
        self.export_csv_rows = None
        self.scan_progress_percentage = 0.0;
        self.genThreeD = False;
        self.theme = "ggplot2"
        self.files_prefix = ""
        self.histograms_title_prefix = ""
        self.dbscan_compare_dict = None;
        self.counter = None
        self.knn = None
        self.epsilons = None
        
        if (is_prescan):
          tar = self.prescan_main
        else:
          tar = self.scan_main
        super().__init__(target=tar)

    def run_prescan(self, experiment_files=[], knn=[], epsilons=[]):
        if (self.is_alive()):
            return False

        self.theme = scanThread.themes["1"]
        self.ex_files = experiment_files
        self.knn = knn
        self.epsilons = epsilons
        self.start()
        return True

    def run_scan(self, experiment_files=[], configuration=[default_configuration], gen_threeD=False, theme="1"):
        if (self.is_alive()):
            return False

        self.ex_files = experiment_files
        self.conf = configuration
        self.genThreeD = gen_threeD
        self.theme = scanThread.themes[theme]
        self.theme_key = theme
        self.start()
        return True 

    def get_scan_status(self):
        return self.status

    def generate_pca_correlation(self):
      try:
        if (self.clusters_df is None or self.original_df is None):
            return None

        get_point = lambda intercept, slope, x: (max(x), (slope * max(x) + intercept)) 
        df = self.clusters_df.query(f"(probe == 0)")
        num_of_points = df.num_of_points.to_numpy()
        num_of_points = [float(n) for n in num_of_points]
        polygon_radius = df.polygon_radius.to_numpy()
        stdev = np.stack(df.pca_std.to_numpy())

        density = num_of_points/(np.pi*stdev[:,0]*stdev[:,1])
        polygon_density = df.polygon_density.to_numpy()

        pca_size = df.pca_size.to_numpy()
        polygon_size = df.polygon_size.to_numpy()
        
        stddev_major = stdev[:,0]

        fig = make_subplots(rows=1, 
                    cols=3,   
                    horizontal_spacing=0.05)

        slope, intercept, r_value, p_value, std_err = stats.linregress(stddev_major,polygon_radius)
        xf, yf = get_point(intercept, slope, stddev_major) 
        fig.add_trace(go.Scattergl(x=stddev_major,
                                   y=polygon_radius,
                                   mode='markers',
                                   marker=dict(color="red", opacity=0.8)),
                      row=1,
                      col=1)

        fig.add_shape(type="line",
                      x0=0,
                      y0=intercept,
                      x1=xf,
                      y1=yf,
                      line=dict(color="blue", width=2),
                      opacity=0.8,
                      row=1,
                      col=1)

        fig.update_xaxes(title_text="PCA major axis", row=1, col=1)
        fig.update_yaxes(title_text="polygon radius (R squared: %f)" % r_value, row=1, col=1)


        slope, intercept, r_value, p_value, std_err = stats.linregress(density,polygon_density)
        xf, yf = get_point(intercept, slope, density) 
        fig.add_trace(go.Scattergl(x=density,
                                   y=polygon_density,
                                   mode='markers',
                                   marker=dict(color="red", opacity=0.8)),
                      row=1,
                      col=2)

        fig.add_shape(type="line",
                      x0=0,
                      y0=intercept,
                      x1=xf,
                      y1=yf,
                      line=dict(color="blue", width=2),
                      opacity=0.8,
                      row=1,
                      col=2)
        fig.update_xaxes(title_text="PCA density", row=1, col=2)
        fig.update_yaxes(title_text="polygon density (R squared: %f)" % r_value, row=1, col=2)

        slope, intercept, r_value, p_value, std_err = stats.linregress(pca_size,polygon_size)
        xf, yf = get_point(intercept, slope, pca_size)
        fig.add_trace(go.Scattergl(x=pca_size,
                                   y=polygon_size,
                                   mode='markers',
                                   marker=dict(color="red", opacity=0.8)),
                      row=1,
                      col=3)   

        fig.add_shape(type="line",
                      x0=0,
                      y0=intercept,
                      x1=xf,
                      y1=yf,
                      line=dict(color="blue", width=2),
                      opacity=0.8,
                      row=1,
                      col=3)
        fig.update_xaxes(title_text="PCA size", row=1, col=3)
        fig.update_yaxes(title_text="polygon size (R squared: %f)" % r_value, row=1, col=3)



        fig.write_html(scan_results_folder + "pca_correlation.html")
        return "pca_correlation.html"    

      except BaseException as be:
        print("WTFFFF")
        print(be)
        format_exception(be)
        raise(be)    

    def generate_db_scan_plot_html(self, conf, original_df_row_index):
        try:
          ind = original_df_row_index

          if (self.clusters_df is None or self.original_df is None):
              return None

          fname = self.original_df.loc[ind].filename
          compare_html = None
          conf_idx = self.conf.index(conf)
          num_of_confs = len(self.conf)
          gen_compare_plot = False

          if (self.dbscan_compare_dict is not None):
            gen_compare_plot = True
            compare_html = fname.replace(".", "_") + "_compare_dbscan.html"

            if self.dbscan_compare_dict.get(fname) == None:
              titles = []
              for sconf in self.conf:
                alg_title = "HDBScan" if sconf["general"]["use_hdbscan"] else "DBScan"
                if (sconf["general"]["use_hdbscan"]):
                  specific_title =  "  %sN(%d), S(%d), Alg(%s), a(%s)%s%sPC(%f)" % ("F(%f) " % sconf["general"]["density_drop_threshold"] if conf["general"]["density_drop_threshold"] > 0.0 else "",
                                                          sconf["hdbscan"]["hdbscan_min_npoints"],\
                                                                                   sconf["hdbscan"]["hdbscan_min_samples"],
                                                                                   sconf["hdbscan"]["hdbscan_extracting_alg"],
                                                                                   sconf["hdbscan"]["hdbscan_alpha"],
                                                                                      (", Eps(%d)" % sconf["hdbscan"]["hdbscan_eps"]) if\
                                                                                        sconf["hdbscan"]["hdbscan_eps"] != -9999 else "",
                                                                                      (", NR STD(%s)" % sconf["general"]["stddev_num"]) if\
                                                                                      sconf["general"]["noise_reduce"] else "",
                                                                                      sconf["general"]["photon_count"])
                else:
                  specific_title = "  %sEps(%d), K(%d)%sPC(%f)" % ("F(%f) " % sconf["general"]["density_drop_threshold"] if conf["general"]["density_drop_threshold"] > 0.0 else "",
                                                          sconf["dbscan"]["dbscan_eps"],\
                                                          sconf["dbscan"]["dbscan_min_samples"],
                                                          (", NR STD(%s)" % sconf["general"]["stddev_num"]) if\
                                                            sconf["general"]["noise_reduce"] else "",
                                                            sconf["general"]["photon_count"])

                title = "%s: %s" % (alg_title, specific_title)
                titles.append(title)

              titles = ["<b>%s</b>" % t for t in titles]          
              self.dbscan_compare_dict[fname] = make_subplots(rows=int((num_of_confs / 2) + (num_of_confs % 2)), 
                                                              cols=2, 
                                                              vertical_spacing=0.1,
                                                              subplot_titles=titles)
          fig = make_subplots(rows=1, 
                              cols=1, 
                              vertical_spacing=0.05,
                              subplot_titles=[fname + "  " + self.histograms_title_prefix])

          # filter clusters found specifically for our working file
          df = self.clusters_df.query(f"(full_path == '{self.original_df.loc[ind].full_path}') and (probe == 0)")
          grey_probe = go.Scattergl(x=self.original_df.loc[ind].probe_0_unassigned['x'].values,
                                    y=self.original_df.loc[ind].probe_0_unassigned['y'].values,
                                    mode='markers',
                                    marker=dict(color=self.theme[1], opacity=0.1))

          # Draw probe 0 in grey 
          fig.add_trace(grey_probe,
                        row=1, 
                        col=1)

          if gen_compare_plot:
            self.dbscan_compare_dict[fname].add_trace(grey_probe,
                                                      row=int((conf_idx / 2) + 1),
                                                      col=int(conf_idx % 2 + 1))

          if self.genThreeD:
            threed_fig = make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]])


          # Draw actual clusters 
          for i in df.index:
              pc = df.loc[i].pointcloud
              x_val = pc['x'].values
              y_val = pc['y'].values
              z_val = pc['z'].values

              if conf["general"]["noise_reduce"]:
                nrc_list = df.loc[i].noise_reduced_clusters.T.tolist()
                x_val = nrc_list[0]
                y_val = nrc_list[1]

                if conf["general"]["use_z"]:
                  z_val = nrc_list[2]

              hover_text = "Cluster #%d <br>" % i
              hover_text += "Number of points: %d<br>" % df.loc[i]['num_of_points']
              hover_text += "Polygon size (XY polygon): %f<br>" % df.loc[i]['polygon_size']
              hover_text += "Polygon perimeter (XY polygon - nm): %f<br>" % df.loc[i]["polygon_perimeter"]
              hover_text += "Polygon radius (XY polygon - nm): %f<br>" % df.loc[i]["polygon_radius"]
              hover_text += "Density (flat polygon): %f<br>" % df.loc[i]['polygon_density']
              if df.loc[i]['reduced_polygon_density'] != -9999: hover_text += "Density (dimension reduced): %f" %  df.loc[i]['reduced_polygon_density']

              clstr = go.Scatter(x=x_val,
                                 y=y_val,
                                 hoverinfo='text',
                                 mode='markers',
                                 marker=dict(color=color_palette[i], opacity=0.8), #color="red", opacity=0.8),
                                 hovertext=hover_text,
                                 text=hover_text)
              if gen_compare_plot:
                self.dbscan_compare_dict[fname].add_trace(clstr,
                                                          row=int((conf_idx / 2) + 1),
                                                          col=int(conf_idx % 2 + 1))
              fig.add_trace(clstr,
                            row=1,
                            col=1)

              for convex_hull_pair in df.loc[i].convex_hull:
                vrt = go.Scattergl(x=convex_hull_pair[:,0],
                                   y=convex_hull_pair[:,1],
                                   mode='lines',
                                   opacity=0.2,
                                   marker=dict(color=color_palette[i], opacity=0.05))#color="red", opacity=0.05))
                fig.add_trace(vrt,
                row=1,
                col=1)

              if self.genThreeD and conf["general"]["use_z"]:
                threed_fig.add_trace(go.Scatter3d(x=pc['x'].values, y=pc['y'].values, z=pc['z'].values, mode='markers', marker=dict(color=color_palette[i])), 1, 1)
          
          fig.update_xaxes(range=[0, 18e3], row=1, col=1)     
          fig.update_layout(template=self.theme[0], height=800, showlegend=False)

          if self.genThreeD:
            threed_fig.update_xaxes(range=[0, 18e3], row=1, col=1)   
            threed_fig.update_layout(template=self.theme[0], height=800, showlegend=False, scene=dict(zaxis=dict(range=[-7e3,7e3])))

          fig_name = self.prefix +\
            fname.replace(".", "_") + "_dbscan_plot.html"

          if self.genThreeD:
            threed_fig.write_html(scan_results_folder + fig_name.replace("_dbscan", "_dbscan_3D"))
          fig.write_html(scan_results_folder + fig_name)

          if gen_compare_plot and conf_idx == (num_of_confs - 1):
            self.dbscan_compare_dict[fname].update_layout(template=self.theme[0], showlegend=False)
            self.dbscan_compare_dict[fname].write_html(scan_results_folder + compare_html)

          return (fig_name, fig_name.replace("_dbscan", "_dbscan_3D") if self.genThreeD else None, compare_html)
        except BaseException as be:
          format_exception(be)
          raise(be)                   

    def generate_global_histograms(self, **kwargs):
      self.prefix = ""
      # In case no groups generated
      if (self.global_clusters_df.empty):
        print("No clusters found at all!")
        # plots
        self.result_json["global_lower_half_num_of_points_html"] = None
        self.result_json["global_upper_halfnum_of_points_html"] = None
        self.result_json["global_num_of_points_html"] = None
        self.result_json["global_lower_half_major_axis_html"] = None
        self.result_json["global_upper_halfmajor_axis_html"] = None
        self.result_json["global_major_axis_html"] = None
        self.result_json["global_lower_half_minor_axis_html"] = None
        self.result_json["global_upper_halfminor_axis_html"] = None
        self.result_json["global_minor_axis_html"] = None
        self.result_json["global_lower_half_density_html"] = None
        self.result_json["global_upper_half_density_html"] = None
        self.result_json["global_density_html"] = None
        self.result_json["global_lower_half_pca_size_html"] = None
        self.result_json["global_upper_halfpca_size_html"] = None
        self.result_json["global_pca_size_html"] = None
        self.result_json["global_lower_half_poly_size_html"] = None
        self.result_json["global_upper_halfpoly_size_html"] = None
        self.result_json["global_poly_size_html"] = None
        self.result_json["global_lower_half_poly_perimeter_html"] = None
        self.result_json["global_upper_halfpoly_perimeter_html"] = None
        self.result_json["global_poly_perimeter_html"] = None
        self.result_json["global_lower_half_poly_radius_html"] = None
        self.result_json["global_upper_halfpoly_radius_html"] = None
        self.result_json["global_poly_radius_html"] = None
        self.result_json["global_poly_size_density_html"] = None
      else:                        
        print("Generating global histograms!")
        df = self.global_clusters_df.query(f"(probe == 0)")
        if ("possible_confs" in kwargs and  len(kwargs["possible_confs"]) == 1):
          histograms_res = self.generate_histograms_html(df, 
                                                         "Average histogram",
                                                         density_drop=kwargs["possible_confs"][0]["general"]["density_drop_threshold"],
                                                         z_density_drop=kwargs["possible_confs"][0]["general"]["threed_drop_threshold"])          
        else:
          histograms_res = self.generate_histograms_html(df, "Average histogram")
        self.result_json["global_lower_half_num_of_points_html"] = histograms_res[0][0]
        self.result_json["global_upper_halfnum_of_points_html"] = histograms_res[0][1]
        self.result_json["global_num_of_points_html"] = histograms_res[0][2]
        self.result_json["global_lower_half_major_axis_html"] = histograms_res[1][0]
        self.result_json["global_upper_halfmajor_axis_html"] = histograms_res[1][1]
        self.result_json["global_major_axis_html"] = histograms_res[1][2]
        self.result_json["global_lower_half_minor_axis_html"] = histograms_res[2][0]
        self.result_json["global_upper_halfminor_axis_html"] = histograms_res[2][1]
        self.result_json["global_minor_axis_html"] = histograms_res[2][2]
        self.result_json["global_lower_half_density_html"] = histograms_res[6][0]
        self.result_json["global_upper_half_density_html"] = histograms_res[6][1]
        self.result_json["global_density_html"] = histograms_res[6][2]
        self.result_json["global_lower_half_pca_size_html"] = histograms_res[4][0]
        self.result_json["global_upper_halfpca_size_html"] = histograms_res[4][1]
        self.result_json["global_pca_size_html"] = histograms_res[4][2]
        self.result_json["global_lower_half_poly_size_html"] = histograms_res[5][0]
        self.result_json["global_upper_halfpoly_size_html"] = histograms_res[5][1]
        self.result_json["global_poly_size_html"] = histograms_res[5][2]            
        self.result_json["global_lower_half_poly_perimeter_html"] = histograms_res[7][0]
        self.result_json["global_upper_halfpoly_perimeter_html"] = histograms_res[7][1]
        self.result_json["global_poly_perimeter_html"] = histograms_res[7][2]
        self.result_json["global_lower_half_poly_radius_html"] = histograms_res[8][0]
        self.result_json["global_upper_halfpoly_radius_html"] = histograms_res[8][1]
        self.result_json["global_poly_radius_html"] = histograms_res[8][2]                  
        self.result_json["global_poly_size_density_html"] = histograms_res[10]

    def generate_histograms_html(self, df, row_title, **kwargs):
        try:
          range_split = lambda min, max, pctg: [(min + (0 if p == 0 else 1) + int(((max - min) * p  )/pctg),\
                                                 min + int(((max - min) * (p + 1)/pctg)))\
                                                 for p in range(0, pctg)]
          float_range_split = lambda min, max, pctg: [(min + ((max - min) * p)/pctg,\
                                                       min + ((max - min) * (p + 1)/pctg))\
                                                       for p in range(0, pctg)]                                               
          count_elements_in_range = lambda r, l: len([i for i in l if i >= r[0] and i <= r[1]])
          count_float_elements_in_range = lambda r, l: len([i for i in l if i >= r[0] and i < r[1]])
          freq_array = lambda p: functools.reduce(lambda a,b: a + b, [[i+1] * v for i,v in enumerate(p)])
          median = lambda a: a[int(len(a) / 2)] if len(a) % 2 == 1\
            else (a[int(len(a) / 2)] + a[int(len(a) / 2) - 1]) * 0.5
          find_item = lambda item, rng: [i for i, a in enumerate(rng) if item >= a[0] and item <= a[1]][0]
          find_float_item = lambda item, rng: [i for i, a in enumerate(rng) if item >= a[0] and item < a[1]][0]
          avg = lambda l: float(sum(l)) / float(len(l))

          #ind = original_df_row_index

          # if (self.clusters_df is None or self.original_df is None):
          #     return None

          # # filter clusters found specifically for our working file
          # df = self.clusters_df.query(f"(full_path == '{self.original_df.loc[ind].full_path}') and (probe == 0)")

          if df is None:
            return None

          num_of_points = df.num_of_points.to_numpy()
          num_of_points = [float(n) for n in num_of_points]
          stdev = np.stack(df.pca_std.to_numpy())
          density = num_of_points/(np.pi*stdev[:,0]*stdev[:,1])
          pca_size = df.pca_size.to_numpy()
          polygon_size = df.polygon_size.to_numpy()
          polygon_density = df.polygon_density.to_numpy()
          reduced_polygon_density = df.reduced_polygon_density.to_numpy()
          polygon_perimeter = df.polygon_perimeter.to_numpy()
          polygon_radius = df.polygon_radius.to_numpy()

          histnames = []

          density_min = 0.0
          density_center = 7.5
          red_density_min = 0.0
          red_density_center = 1.5
          polysize_center=150000.0

          if ("density_drop" in kwargs and kwargs["density_drop"] >= 1.0):
            density_min = 1.0
            polysize_center=150000.0
            density_center = 8.5

          if ("z_density_drop" in kwargs and kwargs["z_density_drop"] >= 1.0):
            red_density_min = 1.0
            red_density_center = 2.5


          hists = [(num_of_points, "Number of points in clusters", "num_of_pts", 0, 225.0, True),
           (stdev[:,0], "Cluster Major axis", "maj_ax_stdev", 0, 50.0, True),
           (stdev[:,1], "Cluster Minor axis", "min_ax_stdev", 0, 50.0, True),
           (density, "Density", "density", 0, 3.0, True),
           (pca_size, "PCA Size", "pca_size", 0, 50, True),
           (polygon_size, "Polygon Size", "poly_size", 0, polysize_center, True),
           (polygon_density, "Polygon Density", "poly_density", density_min, density_center, True),
           (polygon_perimeter, "Polygon Perimeter", "poly_perim", 0, 300.0, True),
           (polygon_radius, "Polygon Radius", "poly_radius", 0, 160.0, True)]


          if reduced_polygon_density[0] != -9999:
            hists.append((reduced_polygon_density, "Reduced Polygon density", "reduced_poly_density", red_density_min, red_density_center, True))



          for hist_data, hist_title, hist_prefix, min_val, center, is_float in hists:
              fig_names = []
              htdt = [float(a) if is_float else int(a) for a in hist_data]
              htdt.sort()
              med = median(htdt)
              mean = avg(htdt)

              # hist_data_and_halfs = [([a for a in htdt if a <= med], "_lower_half", "Probability - Lower half"),
              #                        ([a for a in htdt if a > med], "_upper_half", "Probability - Upper half"),
              #                        (htdt, "", "Probability")]
              # a = hist_data_and_halfs[0][0]
              # a.sort()
              # print("\nLower half:")
              # print(a)
              # a = hist_data_and_halfs[1][0]
              # a.sort()
              # print("\nUpper half:")
              # print(a)        
              # for data, data_prefix, yaxis_title in hist_data_and_halfs:
              #   if len(data) == 0:
              #     fig_names.append(None)
              #     continue

              #   rs =[]
              #   counted_clusters = []
              #   titles = []
              #   fig = make_subplots(rows=1, 
              #                       cols=1, 
              #                       vertical_spacing=0.05,
              #                       row_titles=[self.original_df.loc[ind].filename+\
              #                         self.histograms_title_prefix],
              #                       column_titles=[hist_title])

              #   if is_float:
              #     rs = float_range_split(min(data), max(data) * 1.05 , 20)
              #     titles = ["%f - %f" % (rng[0], rng[1]) for rng in rs]
              #     counted_clusters = [count_float_elements_in_range(rng, data) for rng in rs]
              #   else:
              #     rs = range_split(min(data), max(data) + 1, 20)
              #     titles = ["%d - %d" % (rng[0], rng[1]) for rng in rs]

              #     counted_clusters = [count_elements_in_range(rng, data) for rng in rs]
                  
              #   print("\nRange split for data:")
              #   print(titles)

              #   print("\nCounted data data:")
              #   print(counted_clusters)

              #   fig.add_trace(go.Histogram(histfunc="sum", histnorm="probability", y=counted_clusters, x=titles, name="sum"))
              #   if data_prefix is "":
              #     if is_float:
              #       median_idx = find_float_item(med, rs)
              #     else:
              #       print("Median: ", med)
              #       print("Rs: ", rs)
              #       median_idx = find_item(med, rs)
              #     if is_float:
              #       mean_idx = find_float_item(mean, rs)
              #     else:
              #       mean_idx = find_item(int(mean), rs)

              #     mean_coeff = 1
              #     med_coeff = -1
              #     if mean_idx < median_idx:
              #       mean_coeff = -1
              #       med_coeff = 1

              #     height = max(counted_clusters) / sum(counted_clusters) + 0.05
              #     fig.add_shape(type="line",
              #                   x0=median_idx,
              #                   y0=0,
              #                   x1=median_idx,
              #                   y1=height,
              #                   line=dict(color=self.theme[2], width=2, dash="dot"),
              #                   opacity=1)

              #     fig.add_annotation(x=median_idx,
              #                        y=height,
              #                        text=("Median = %f" if is_float else "Median = %d") % med ,
              #                        showarrow=False,
              #                        xshift= med_coeff * 65 if is_float else med_coeff * 50,
              #                        yshift=-10)

                  
              #     fig.add_shape(type="line",
              #                   x0=mean_idx,
              #                   y0=0,
              #                   x1=mean_idx,
              #                   y1=height,
              #                   line=dict(color=self.theme[2], width=2, dash="dot"),
              #                   opacity=1)

              #     fig.add_annotation(x=mean_idx,
              #                        y=height,
              #                        text=("Mean = %f" if is_float else "Mean = %d") % mean,
              #                        showarrow=False,
              #                        xshift= mean_coeff * 60 if is_float else mean_coeff * 45,
              #                        yshift=-10)              

              #   fig.update_layout(template=self.theme[0],
              #                     xaxis_title_text='Range',
              #                     yaxis_title_text=yaxis_title + " - (Total: %d)" % sum(counted_clusters),
              #                     bargap=0.015)

              #   fig_name = self.prefix +\
              #     self.original_df.loc[ind].filename.replace(".", "_") +\
              #     ("%s_%s.html" % (data_prefix,hist_prefix)) 
              #   fig.write_html(scan_results_folder + fig_name)
              #   fig_names.append(fig_name)
              # histnames.append(fig_names)

              n_bins = 15
              bin_size = (center - min_val)/float(n_bins)
              upper_range = [(center + i * (center - min_val)/float(n_bins), center + (i + 1) * (center - min_val)/float(n_bins))   for i in range(0, n_bins)]
              lower_range = [(min_val + i * (center - min_val)/float(n_bins), min_val + (i + 1) * (center - min_val)/float(n_bins))   for i in range(0, n_bins)]
              total_range = lower_range + upper_range

              print("\n\n\n\n\n\n%s %f" % (hist_prefix, float(med)))
              print("\nTotal data:")
              print(htdt)
              hist_data_and_halfs = [([a for a in htdt if a < center], "_lower_half", "Probability - Lower half", lower_range, False),
                                     ([a for a in htdt if a >= center], "_upper_half", "Probability - Upper half", upper_range, False),
                                     (htdt, "", "Probability", total_range, True)]

              a = hist_data_and_halfs[0][0]
              a.sort()
              print("\nLower half:")
              print(a)
              a = hist_data_and_halfs[1][0]
              a.sort()
              print("\nUpper half:")
              print(a)        
              for data, data_prefix, yaxis_title, rs, is_total_hist in hist_data_and_halfs:
                if len(data) == 0:
                  fig_names.append(None)
                  continue

                counted_clusters = []
                titles = []
                fig = make_subplots(rows=1, 
                                    cols=1, 
                                    vertical_spacing=0.05,
                                    row_titles=[row_title],
                                    column_titles=[hist_title])

                if is_float:
                  titles = ["%f - %f" % (rng[0], rng[1]) for rng in rs]

                  if is_total_hist:
                    rs[-1] = (rs[-1][0], sorted(data)[-1] + 1.0)
                    titles[-1] = ">%f" % rs[-1][0]

                  counted_clusters = [count_float_elements_in_range(rng, data) for rng in rs]

                else:
                  titles = ["%d - %d" % (rng[0], rng[1]) for rng in rs]

                  if is_total_hist:
                    rs[-1] = (rs[-1][0], sorted(data)[-1] + 1)
                    titles[-1] = ">%d" % rs[-1][0]

                  counted_clusters = [count_elements_in_range(rng, data) for rng in rs]

                  
                print("\nRange split for data:")
                print(titles)

                print("\nCounted data data %f:" % bin_size)
                print(counted_clusters)

                mn_bin = rs[0][0] 
                mx_bin = rs[-1][1]                
                fig.add_trace(go.Histogram(histfunc="sum", histnorm="probability density", xbins=dict(start=mn_bin, end=mx_bin), y=counted_clusters, x=titles, name="sum"))
                fig.update_layout(template=self.theme[0],
                                  xaxis_title_text='Range',
                                  yaxis_title_text=yaxis_title + " - (Total: %d)" % sum(counted_clusters),
                                  bargap=0.0075)
                if is_total_hist:
                  fig.update_xaxes(range=[0,30])
                else: 
                  fig.update_xaxes(range=[0,15])

                fig_name = self.prefix + row_title +\
                  ("%s_%s.html" % (data_prefix,hist_prefix)) 
                fig.write_html(scan_results_folder + fig_name)
                fig_names.append(fig_name)
              histnames.append(fig_names)


          fig = make_subplots(rows=1, 
                cols=1, 
                vertical_spacing=0.05)

          fig.add_trace(go.Scattergl(x=polygon_size,
                                     y=polygon_density,
                                     mode='markers',
                                     marker=dict(color='purple', opacity=1)),
                        row=1, 
                        col=1)

          fig.update_layout(template=self.theme[0],
                            xaxis_title_text='Polygon size',
                            yaxis_title_text='Polygon density')

          different_name = self.prefix + row_title +\
            ("%s.html" % ("size_in_dependence_of_density")) 

          fig.write_html(scan_results_folder + different_name)
          histnames.append(different_name)              
          return histnames
        except BaseException as be:
          format_exception(be)
          raise(be)             

    def generate_probe_visualization_plot_html(self, conf, original_df_row_index):

        ind = original_df_row_index

        if (self.original_df is None):
            return None

        fig = make_subplots(rows=1, 
                            cols=1, 
                            specs=[[{'is_3d': True}]],
                            vertical_spacing=0.05,
                            subplot_titles=[self.original_df.loc[ind].filename+\
                              self.histograms_title_prefix])
            
        # Draw probe 0 in red 
        # threed_fig.add_trace(go.Scatter3d(x=pc['x'].values, y=pc['y'].values, z=pc['z'].values), 1, 1)
        fig.add_trace(go.Scatter3d(x=self.original_df.loc[ind].pointcloud.query('probe == 0')['x'].values,
                                   y=self.original_df.loc[ind].pointcloud.query('probe == 0')['y'].values,
                                   z=self.original_df.loc[ind].pointcloud.query('probe == 0')['z'].values,
                                   mode='markers',
                                   marker=dict(color='red', opacity=1)),
                      row=1, 
                      col=1)

        # fig.add_trace(go.Scattergl(x=self.original_df.loc[ind].pointcloud.query('probe == 1')['x'].values,
        #                            y=self.original_df.loc[ind].pointcloud.query('probe == 1')['y'].values,
        #                            mode='markers',
        #                            marker=dict(color='limegreen', opacity=1)),
        #               row=1, 
        #               col=1)

        fig.update_xaxes(range=[0, 18e3], row=1, col=1) 
        #fig.update_zaxes(range=[-8e3, 8e3], row=1, col=1)    
        fig.update_layout(template=self.theme[0], height=800, showlegend=False, scene=dict(zaxis=dict(range=[-7e3,7e3])))
        fig_name = self.prefix +\
          self.original_df.loc[original_df_row_index].filename.replace(".", "_") + "_probevis_plot.html"
        fig.write_html(scan_results_folder + fig_name)

        return fig_name

    def generate_kdist_plot_html(self, conf, original_df_row_index, num_of_neighbours=[4, 8, 16, 32, 64], in_pc=None, file_name=None, results_folder=None):
        try: 
          results_folder = scan_results_folder if results_folder is None else results_folder
          ind = original_df_row_index
          fig = make_subplots(rows=int((len(num_of_neighbours) / 2) + (len(num_of_neighbours) % 2)), 
                              cols=2 if len(num_of_neighbours) > 1 else 1,  
                              vertical_spacing=0.13,
                              subplot_titles=["Distance from k(%d) neighbor per localization" % d for d in num_of_neighbours])

          pc = self.original_df.loc[ind].pointcloud.query("probe == 0") if in_pc is None else in_pc
          fn = self.original_df.loc[original_df_row_index].filename.replace(".", "_") if file_name is None else file_name
              
          if conf["k-dist"]["is_3D"]:
              points = pc[['x', 'y' ,'z']].to_numpy()
          else:
              points = pc[['x', 'y']].to_numpy()
          
          nbrs = NearestNeighbors(n_neighbors=conf["k-dist"]["n_neighbors"],
                                  algorithm=conf["k-dist"]["algorithm"]).fit(points)
          distances, _ = nbrs.kneighbors(points)

          for i,k in enumerate(num_of_neighbours):
            t = [p[k] for p in distances]
            t.sort()
            nrm_xval = np.arange(len(t))
            xval = [(float(x) * 100) / (len(t)-1) for x in nrm_xval]
            fig.add_trace(go.Scattergl(x=xval,
                                       y=t,
                                       mode='lines+markers',
                                       marker=dict(opacity=1)),
                          row=int((i / 2) + 1),
                          col=int(i % 2 + 1))
            fig.update_xaxes(title_text="%% of localizations", row=int((i / 2) + 1), col=int(i % 2 + 1))
            fig.update_yaxes(range=[0,max(t)], title_text="Distance (nm)", row=int((i / 2) + 1), col=int(i % 2 + 1))

            arr = [(nrm_xval[i], t[i]) for i in range(0, len(t))]
            pt = [nrm_xval[-1], t[0]]
            dist_arr = [(np.sqrt((pt[0] - x) ** 2 + (pt[1] - y) ** 2)) for (x,y) in arr]
            knee_point = (xval[dist_arr.index(min(dist_arr))], t[dist_arr.index(min(dist_arr))])
            fig.add_trace(go.Scattergl(x=[knee_point[0]],
                                       y=[knee_point[1]],
                                       mode='markers',
                                       marker=dict(opacity=1, size=15, color='LightSkyBlue')),
                          row=int((i / 2) + 1),
                          col=int(i % 2 + 1))
          fig.update_layout(template=self.theme[0], height=800, showlegend=False)
          fig_name = self.prefix + fn + "_kdist_plot.html"
          fig.write_html(results_folder + fig_name)
          
          return fig_name
        except BaseException as be:
          format_exception(be)
          raise(be) 

    def generate_epsdist_plot_html(self, conf, original_df_row_index, epsilons=[50, 100], in_pc=None, file_name=None, results_folder=None):
        try: 
          results_folder = scan_results_folder if results_folder is None else results_folder
          ind = original_df_row_index
          fig = make_subplots(rows=int((len(epsilons) / 2) + (len(epsilons) % 2)), 
                              cols=2 if len(epsilons) > 1 else 1,  
                              vertical_spacing=0.13,
                              subplot_titles=["Numbers of neighbours in eps(%d) neighborhood per localization" % d for d in epsilons])

          pc = self.original_df.loc[ind].pointcloud.query("probe == 0") if in_pc is None else in_pc
          fn = self.original_df.loc[original_df_row_index].filename.replace(".", "_") if file_name is None else file_name
              
          if conf["k-dist"]["is_3D"]:
              points = pc[['x', 'y' ,'z']].to_numpy()
          else:
              points = pc[['x', 'y']].to_numpy()
          
          nbrs = NearestNeighbors().fit(points)

          for i,eps in enumerate(epsilons):
            res = nbrs.radius_neighbors(X=points, radius=eps)
            t = [len(a) for a in res[1]] 
            t.sort(reverse=True)
            
            nrm_xval = np.arange(len(t))
            xval = [(float(x) * 100) / (len(t)-1) for x in nrm_xval]
            fig.add_trace(go.Scattergl(x=xval,
                                       y=t,
                                       mode='lines+markers',
                                       marker=dict(opacity=1)),
                          row=int((i / 2) + 1),
                          col=int(i % 2 + 1))
            fig.update_xaxes(title_text="% Localizations", row=int((i / 2) + 1), col=int(i % 2 + 1))
            fig.update_yaxes(title_text="Number of neighbors in neighborhood", row=int((i / 2) + 1), col=int(i % 2 + 1))

            arr = [(xval[i], t[i]) for i in range(0, len(t))]
            pt = [xval[0], t[-1]]
            print(pt)
            dist_arr = [(np.sqrt((pt[0] - x) ** 2 + (pt[1] - y) ** 2)) for (x,y) in arr]
            knee_point = (xval[dist_arr.index(min(dist_arr))], t[dist_arr.index(min(dist_arr))])
            fig.add_trace(go.Scattergl(x=[knee_point[0]],
                                       y=[knee_point[1]],
                                       mode='markers',
                                       marker=dict(opacity=1, size=15, color='LightSkyBlue')),
                          row=int((i / 2) + 1),
                          col=int(i % 2 + 1))

          fig.update_layout(template=self.theme[0], height=800, showlegend=False)
          fig_name = self.prefix + fn + "_epsdist_plot.html"
          fig.write_html(results_folder + fig_name)
          
          return fig_name
        except BaseException as be:
          format_exception(be)
          raise(be) 


    def generate_output_from_scan(self, conf, dataset_result):
      print("Generating ouptut of scan")

      self.temp_result_holder = []
      dataset = dataset_result
      self.original_df = dataset.orig_df
      self.clusters_df = dataset.groups_df
      self.global_df = pd.concat([self.global_df, dataset.orig_df])
      self.result_json["num_of_files"] = 0 
      if self.clusters_df is not None:
        self.global_clusters_df = pd.concat([self.global_clusters_df, dataset.groups_df])
        self.clusters_df['pca_major_axis_std'] = self.clusters_df['pca_std'].apply(lambda x: x[0])
        self.clusters_df['pca_minor_axis_std'] = self.clusters_df['pca_std'].apply(lambda x: x[1])

      for i in self.original_df.index:
          file_res = {}
          # general information
          file_res["filename"] = self.prefix + self.original_df.loc[i].filename
          file_res["titlename"] = self.original_df.loc[i].filename
          file_res["coprotein"] = self.original_df.loc[i].coprotein
          file_res["label"] = self.original_df.loc[i].label
          file_res["probe0_num_of_points"] = int(self.original_df.loc[i].probe0_num_of_points)
          file_res["probe0_ngroups"] = int(self.original_df.loc[i].probe0_ngroups)
          file_res["probe1_num_of_points"] = int(self.original_df.loc[i].probe1_num_of_points)
          file_res["probe0_area"] = int(self.original_df.loc[i].probe0_area)
          file_res["probe0_cluster_density"] = float(self.original_df.loc[i].probe0_cluster_density)

          try:
            file_res["hdbscan"] = conf["general"]["use_hdbscan"]
            file_res["use_z"] = conf["general"]["use_z"]
            file_res["noise_reduced"] = conf["general"]["noise_reduce"]
            file_res["stddev_num"] = conf["general"]["stddev_num"]
            file_res["alg_param_1"] = conf["hdbscan"]["hdbscan_min_npoints"] if file_res["hdbscan"] else conf["dbscan"]["dbscan_eps"]
            file_res["alg_param_2"] = conf["hdbscan"]["hdbscan_min_samples"] if file_res["hdbscan"] else conf["dbscan"]["dbscan_min_samples"]
            file_res["alg_param_3"] = conf["hdbscan"]["hdbscan_eps"] if file_res["hdbscan"] else None
            file_res["alg_param_4"] = conf["hdbscan"]["hdbscan_extracting_alg"] if file_res["hdbscan"] else None
            file_res["alg_param_5"] = conf["hdbscan"]["hdbscan_alpha"] if file_res["hdbscan"] else None

            if (file_res["probe1_num_of_points"] > 0):
                file_res["probe1_ngroups"] = int(self.original_df.loc[i].probe1_ngroups)

            # probe 0 information
            if self.clusters_df is not None:
              q = "(filename == \"%s\") and (probe == 0)" % self.original_df.loc[i].filename
              df = self.clusters_df.query(q).copy()
              agg_df = df.groupby('filename').agg({'num_of_points': ['sum', 'mean', 'median'],
                                                   'pca_major_axis_std': ['mean', 'median'],
                                                   'pca_minor_axis_std': ['mean', 'median'],
                                                   'pca_size': ['mean', 'median'],
                                                   'polygon_size' : ['mean', 'median'],
                                                   'polygon_density' : ['mean', 'median'],
                                                   'reduced_polygon_density': ['mean', 'median'],
                                                   'polygon_radius' : ['mean', 'median'],
                                                   'polygon_perimeter' : ['mean', 'median']}).reset_index()

            file_res["probe_0"] = { "num_of_points":                                      
                                        {"total" :
                                            int(self.original_df.loc[i].probe0_num_of_points),
                                         "cluster_sum" :
                                            int(agg_df["num_of_points"]["sum"][0]\
                                              if file_res["probe0_ngroups"] > 0 else 0),
                                         "mean" :    
                                            "{:.4f}".format(float(agg_df["num_of_points"]["mean"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0),
                                         "median" :  
                                            "{:.4f}".format(float(agg_df["num_of_points"]["median"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0)},
                                   "pca_major_axis_std" : 
                                        {"mean" :    
                                            "{:.4f}".format(float(agg_df["pca_major_axis_std"]["mean"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0),
                                         "median" :  
                                            "{:.4f}".format(float(agg_df["pca_major_axis_std"]["median"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0)},
                                   "pca_minor_axis_std" : 
                                        {"mean" :    
                                            "{:.4f}".format(float(agg_df["pca_minor_axis_std"]["mean"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0),
                                         "median" :  
                                            "{:.4f}".format(float(agg_df["pca_minor_axis_std"]["median"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0)},
                                   "pca_size" : 
                                        {"mean" :    
                                            "{:.4f}".format(float(agg_df["pca_size"]["mean"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0),
                                         "median" :  
                                            "{:.4f}".format(float(agg_df["pca_size"]["median"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0)},
                                   "polygon_size" : 
                                        {"mean" :    
                                            "{:.4f}".format(float(agg_df["polygon_size"]["mean"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0),
                                         "median" :  
                                            "{:.4f}".format(float(agg_df["polygon_size"]["median"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0)},
                                   "polygon_density" : 
                                        {"mean" :    
                                            "{:.4f}".format(float(agg_df["polygon_density"]["mean"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0),
                                         "median" :  
                                            "{:.4f}".format(float(agg_df["polygon_density"]["median"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0)},
                                   "reduced_polygon_density" : 
                                        {"mean" :    
                                            "{:.4f}".format(float(agg_df["reduced_polygon_density"]["mean"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0),
                                         "median" :  
                                            "{:.4f}".format(float(agg_df["reduced_polygon_density"]["median"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0)},
                                   "polygon_perimeter" : 
                                        {"mean" :    
                                            "{:.4f}".format(float(agg_df["polygon_perimeter"]["mean"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0),
                                         "median" :  
                                            "{:.4f}".format(float(agg_df["polygon_perimeter"]["median"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0)},
                                   "polygon_radius" : 
                                        {"mean" :    
                                            "{:.4f}".format(float(agg_df["polygon_radius"]["mean"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0),
                                         "median" :  
                                            "{:.4f}".format(float(agg_df["polygon_radius"]["median"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0)} }

              # probe 1 information
            if file_res["probe1_num_of_points"] > 0:
              if self.clusters_df is not None:
                q = "(filename == \"%s\") and (probe == 1)" % self.original_df.loc[i].filename
                df = self.clusters_df.query(q).copy()
                agg_df = df.groupby('filename').agg({'num_of_points': ['sum', 'mean', 'median'],
                                                     'pca_major_axis_std': ['mean', 'median'],
                                                     'pca_minor_axis_std': ['mean', 'median'],
                                                     'pca_size': ['mean', 'median']}).reset_index()

              file_res["probe_1"] = {"num_of_points" :                                      
                                      {"total":
                                          int(self.original_df.loc[i].probe0_num_of_points)\
                                          if file_res["probe0_ngroups"] > 0 else 0,
                                       "cluster_sum":
                                          int(agg_df["num_of_points"]["sum"][0])\
                                          if file_res["probe0_ngroups"] > 0 else 0,
                                       "mean" :    
                                          "{:.4f}".format(float(agg_df["num_of_points"]["mean"][0]))\
                                          if file_res["probe0_ngroups"] > 0 else 0,
                                       "median" :  
                                          "{:.4f}".format(float(agg_df["num_of_points"]["median"][0]))}\
                                          if file_res["probe0_ngroups"] > 0 else 0,
                                     "pca_major_axis_std" : 
                                          {"mean" :    
                                              "{:.4f}".format(float(agg_df["pca_major_axis_std"]["mean"][0]))\
                                              if file_res["probe0_ngroups"] > 0 else 0,
                                           "median" :  
                                              "{:.4f}".format(float(agg_df["pca_major_axis_std"]["median"][0]))}\
                                              if file_res["probe0_ngroups"] > 0 else 0,
                                     "pca_minor_axis_std" : 
                                          {"mean" :    
                                              "{:.4f}".format(float(agg_df["pca_minor_axis_std"]["mean"][0]))\
                                              if file_res["probe0_ngroups"] > 0 else 0,
                                           "median" :  
                                              "{:.4f}".format(float(agg_df["pca_minor_axis_std"]["median"][0]))}\
                                              if file_res["probe0_ngroups"] > 0 else 0,
                                     "pca_size" : 
                                          {"mean" :    
                                              "{:.4f}".format(float(agg_df["pca_size"]["mean"][0]))\
                                              if file_res["probe0_ngroups"] > 0 else 0,
                                           "median" :  
                                              "{:.4f}".format(float(agg_df["pca_size"]["median"][0])\
                                              if file_res["probe0_ngroups"] > 0 else 0)}}
            # In case no groups generated
            if (file_res["probe0_ngroups"] <= 0):
              # plots
              file_res["dbscan_html"] = None
              file_res["threeD_dbscan_html"] =  None
              file_res["dbscan_compare_html"] = None
              file_res["probevis_html"] = None
              file_res["kdist_html"] = None
              file_res["lower_half_num_of_points_html"] = None
              file_res["upper_halfnum_of_points_html"] = None
              file_res["num_of_points_html"] = None
              file_res["lower_half_major_axis_html"] = None
              file_res["upper_halfmajor_axis_html"] = None
              file_res["major_axis_html"] = None
              file_res["lower_half_minor_axis_html"] = None
              file_res["upper_halfminor_axis_html"] = None
              file_res["minor_axis_html"] = None
              file_res["lower_half_density_html"] = None
              file_res["upper_half_density_html"] = None
              file_res["density_html"] = None
              file_res["lower_half_pca_size_html"] = None
              file_res["upper_halfpca_size_html"] = None
              file_res["pca_size_html"] = None
              file_res["lower_half_poly_size_html"] = None
              file_res["upper_halfpoly_size_html"] = None
              file_res["poly_size_html"] = None
              file_res["lower_half_poly_perimeter_html"] = None
              file_res["upper_halfpoly_perimeter_html"] = None
              file_res["poly_perimeter_html"] = None
              file_res["lower_half_poly_radius_html"] = None
              file_res["upper_halfpoly_radius_html"] = None
              file_res["poly_radius_html"] = None
            else:          
              # plots
              dbscan_res = self.generate_db_scan_plot_html(conf, i)
              file_res["dbscan_html"] = dbscan_res[0]
              file_res["threeD_dbscan_html"] = dbscan_res[1]
              file_res["dbscan_compare_html"] = dbscan_res[2]
                      
              file_res["probevis_html"] = self.generate_probe_visualization_plot_html(conf, i)
              file_res["kdist_html"] = self.generate_kdist_plot_html(conf, i)
              
              # # filter clusters found specifically for our working file
              df = self.clusters_df.query(f"(full_path == '{self.original_df.loc[i].full_path}') and (probe == 0)")
              histograms_res = self.generate_histograms_html(df,
                                                             self.original_df.loc[i].filename.replace(".", "_"),
                                                             density_drop=conf["general"]["density_drop_threshold"],
                                                             z_density_drop=conf["general"]["threed_drop_threshold"])

              file_res["lower_half_num_of_points_html"] = histograms_res[0][0]
              file_res["upper_halfnum_of_points_html"] = histograms_res[0][1]
              file_res["num_of_points_html"] = histograms_res[0][2]
              file_res["lower_half_major_axis_html"] = histograms_res[1][0]
              file_res["upper_halfmajor_axis_html"] = histograms_res[1][1]
              file_res["major_axis_html"] = histograms_res[1][2]
              file_res["lower_half_minor_axis_html"] = histograms_res[2][0]
              file_res["upper_halfminor_axis_html"] = histograms_res[2][1]
              file_res["minor_axis_html"] = histograms_res[2][2]
              file_res["lower_half_density_html"] = histograms_res[6][0]
              file_res["upper_half_density_html"] = histograms_res[6][1]
              file_res["density_html"] = histograms_res[6][2]
              file_res["lower_half_pca_size_html"] = histograms_res[4][0]
              file_res["upper_halfpca_size_html"] = histograms_res[4][1]
              file_res["pca_size_html"] = histograms_res[4][2]
              file_res["lower_half_poly_size_html"] = histograms_res[5][0]
              file_res["upper_halfpoly_size_html"] = histograms_res[5][1]
              file_res["poly_size_html"] = histograms_res[5][2]            
              file_res["lower_half_poly_perimeter_html"] = histograms_res[7][0]
              file_res["upper_halfpoly_perimeter_html"] = histograms_res[7][1]
              file_res["poly_perimeter_html"] = histograms_res[7][2]
              file_res["lower_half_poly_radius_html"] = histograms_res[8][0]
              file_res["upper_halfpoly_radius_html"] = histograms_res[8][1]
              file_res["poly_radius_html"] = histograms_res[8][2]            

            file_res["completed"] = True
          except BaseException as be:
            file_res["completed"] = False
            self.result_json["errors"] = 1      
            format_exception(be)

          self.temp_result_holder.append(file_res)
          self.result_json["file_results"].append(file_res)
          self.result_json["file_results"].sort(key=lambda f: f["titlename"])
          if (self.result_json["pca_correlation"] == False):
            self.result_json["pca_correlation"] = True
            self.result_json["pca_correlation_dir"] = self.generate_pca_correlation()
          self.scan_progress_percentage = self.counter / (len(self.original_df.index) * len(self.conf))
          self.counter += 1.0
      self.result_json["num_of_files"] += len(self.original_df.index)

    def generate_export_csv_row(self, conf):
      for file_res in self.temp_result_holder:
        print("Exporting %s" % file_res["titlename"])

        row = [file_res["titlename"],\
               file_res["probe0_ngroups"],
               file_res["probe_0"]["num_of_points"]["total"],
               file_res["probe_0"]["num_of_points"]["cluster_sum"],
               file_res["probe_0"]["num_of_points"]["mean"],
               file_res["probe_0"]["num_of_points"]["median"],
               file_res["probe_0"]["polygon_size"]["mean"],
               file_res["probe_0"]["polygon_size"]["median"],
               file_res["probe_0"]["polygon_density"]["mean"],
               file_res["probe_0"]["polygon_density"]["median"],
               file_res["probe_0"]["reduced_polygon_density"]["mean"],
               file_res["probe_0"]["reduced_polygon_density"]["median"],
               file_res["probe_0"]["pca_size"]["mean"],
               file_res["probe_0"]["pca_size"]["median"],
               file_res["probe_0"]["pca_major_axis_std"]["mean"],
               file_res["probe_0"]["pca_major_axis_std"]["median"],
               file_res["probe_0"]["pca_minor_axis_std"]["mean"],
               file_res["probe_0"]["pca_minor_axis_std"]["median"],
               "HDBscan" if conf["general"]["use_hdbscan"] else "DBScan",
               "True" if conf["general"]["use_z"] else "False",
               "True" if conf["general"]["noise_reduce"] else "False",
               conf["general"]["stddev_num"] if conf["general"]["noise_reduce"] else "N/A",
               conf["general"]["density_drop_threshold"] if conf["general"]["density_drop_threshold"] != 0.0 else "-",
               conf["general"]["threed_drop_threshold"] if conf["general"]["threed_drop_threshold"] != 0.0 else "-",
               conf["hdbscan"]["hdbscan_min_npoints"] if conf["general"]["use_hdbscan"] else "N/A",
               conf["hdbscan"]["hdbscan_min_samples"] if conf["general"]["use_hdbscan"] else "N/A",
               conf["hdbscan"]["hdbscan_eps"] if conf["hdbscan"]["hdbscan_eps"] != -9999 else "-",
               conf["hdbscan"]["hdbscan_alpha"] if conf["general"]["use_hdbscan"] else "N/A",
               conf["hdbscan"]["hdbscan_extracting_alg"] if conf["general"]["use_hdbscan"] else "N/A",
               conf["dbscan"]["dbscan_min_samples"] if not conf["general"]["use_hdbscan"] else "N/A",               
               conf["dbscan"]["dbscan_eps"] if not conf["general"]["use_hdbscan"] else "N/A",
               conf["dbscan"]["min_npoints"] if not conf["general"]["use_hdbscan"] else "N/A"]

        self.export_csv_rows.append(row)

    def scan_main(self):
        try:
          self.status = scanThread.scanStatus.SCAN_RUNNING
          self.result_json["file_results"] = []
          self.counter = 1.0

          rmtree(scan_results_folder, ignore_errors=True)
          os.mkdir(scan_results_folder)

          rmtree(export_folder)
          os.mkdir(export_folder)
          self.result_json["errors"] = 0

          if (len(self.conf) > 1):
            self.dbscan_compare_dict = {}

          self.export_csv_rows = [["Filename",\
                                  "Number of clusters",
                                  "Number of localizations",
                                  "Number of localizations assigned to clusters",
                                  "Mean localizations per cluster",
                                  "Median localizations per cluster",
                                  "Polygon surronding flat cluster mean size (nm)",
                                  "Polygon surronding flat cluster median size (nm)",
                                  "Polygon mean density",
                                  "Polygon median density",
                                  "Reduced polygon (Including Z-Axis) mean density",
                                  "Reduced polygon (Including Z-Axis) median density",
                                  "PCA mean size",
                                  "PCA median size",
                                  "PCA major axis (per cluster) mean std",
                                  "PCA major axis (per cluster) median std",
                                  "PCA minor axis (per cluster) mean std",
                                  "PCA minor axis (per cluster) median std",
                                  "Algorithm used",
                                  "Z-Axis included"
                                  "Noise reduction",
                                  "Standard score for noise reduction",
                                  "Density threshold",
                                  "Z-axis included desnity threshold",
                                  "HDBscan Min num of points (K)",
                                  "HDBscan Min samples",
                                  "HDBscan Epsilon", 
                                  "HDBscan Alpha",
                                  "HDBscan Extracting alg",
                                  "DBscan Min samples (K)",
                                  "DBscan Epsilon",
                                  "DBscan Min Num of points"]]
            # run DBScan
          for conf in self.conf:
            self.prefix = make_prefix(conf)
            self.histograms_title_prefix = ""
            
            # if conf["general"]["use_hdbscan"] is False:            
            #   self.histograms_title_prefix = "%s Eps(%d), K(%d)%s" % ("F%f" % conf["general"]["density_drop_threshold"] if conf["general"]["density_drop_threshold"] > 0.0 else "",
            #                                                         conf["dbscan"]["dbscan_eps"],\
            #                                                         conf["dbscan"]["dbscan_min_samples"],
            #                                                         (", NR STD(%s)" % conf["general"]["stddev_num"]) if\
            #                                                             conf["general"]["noise_reduce"] else "")         
            # else:             
            #   self.histograms_title_prefix = "%s N(%d), S(%d), Alg(%s), a(%s)%s%s" % ("F%f" % conf["general"]["density_drop_threshold"] if conf["general"]["density_drop_threshold"] > 0.0 else "",
            #                                                                         conf["hdbscan"]["hdbscan_min_npoints"],\
            #                                                                         conf["hdbscan"]["hdbscan_min_samples"],
            #                                                                         conf["hdbscan"]["hdbscan_extracting_alg"],
            #                                                                         conf["hdbscan"]["hdbscan_alpha"],
            #                                                                         (", Eps(%d)" % conf["hdbscan"]["hdbscan_eps"]) if\
            #                                                                           conf["hdbscan"]["hdbscan_eps"] != -9999 else "",
            #                                                                         (", NR STD(%s)" % conf["general"]["stddev_num"]) if\
            #                                                                           conf["general"]["noise_reduce"] else "")
            dataset = DstormDatasetDBSCAN(root=self.ex_files, 
                                          min_npoints=conf["dbscan"]["min_npoints"], 
                                          dbscan_eps=conf["dbscan"]["dbscan_eps"],
                                          dbscan_min_samples=conf["dbscan"]["dbscan_min_samples"],
                                          coloc_distance=conf["coloc"]["coloc_distance"],
                                          coloc_neighbors=conf["coloc"]["coloc_neighbors"],
                                          workers=conf["coloc"]["workers"],
                                          use_hdbscan=conf["general"]["use_hdbscan"],
                                          use_z=conf["general"]["use_z"],
                                          noise_reduce=conf["general"]["noise_reduce"],
                                          stddev_num=conf["general"]["stddev_num"],
                                          hdbscan_min_npoints=conf["hdbscan"]["hdbscan_min_npoints"],
                                          hdbscan_min_samples=conf["hdbscan"]["hdbscan_min_samples"],
                                          hdbscan_epsilon_threshold=conf["hdbscan"]["hdbscan_eps"],
                                          hdbscan_extracting_alg=conf["hdbscan"]["hdbscan_extracting_alg"],
                                          hdbscan_alpha=conf["hdbscan"]["hdbscan_alpha"],
                                          density_drop_threshold=conf["general"]["density_drop_threshold"],
                                          z_density_drop_threshold=conf["general"]["threed_drop_threshold"],
                                          photon_count=conf["general"]["photon_count"])
            print("Completed storm scan, generating results.")
            self.result_json["pca_correlation"] = False;
            self.generate_output_from_scan(conf, dataset)
            self.generate_export_csv_row(conf)



          # generate csv
          with open('%s/%s' % (export_folder, export_file), 'w', newline='') as file:
              wrt = writer(file)
              wrt.writerows(self.export_csv_rows)

          self.generate_global_histograms(possible_confs=self.conf)
          self.status = scanThread.scanStatus.SCAN_FINISHED_SUCCESSFULLY

        except BaseException as be:
          format_exception(be)
          print("Fatal error here!\n")
          self.result_json["errors"] = 2

    def prescan_main(self):
        self.status = scanThread.scanStatus.SCAN_RUNNING
        self.result_json["file_results"] = []
        self.counter = 1.0

        rmtree(prescan_results_folder, ignore_errors=True)
        os.mkdir(prescan_results_folder)
        self.result_json["errors"] = 0

        self.prefix = ""
        try:
          for file in self.ex_files:
            file_res = {}
            filename = file.replace(".", "_");
            filename = filename[filename.rfind("/") + 1:]
            file_res["filename"] = filename
            print("Running for file: %s" % filename)
            file_res["kdist_html"] = self.generate_kdist_plot_html(conf=self.default_configuration, 
                                                                   original_df_row_index=0,
                                                                   num_of_neighbours=self.knn,
                                                                   in_pc=pd.read_csv(file), 
                                                                   file_name=filename,
                                                                   results_folder=prescan_results_folder)
            file_res["epsdist_html"] = self.generate_epsdist_plot_html(conf=self.default_configuration, 
                                                                       original_df_row_index=0,
                                                                       epsilons=self.epsilons,
                                                                       in_pc=pd.read_csv(file), 
                                                                       file_name=filename,
                                                                       results_folder=prescan_results_folder)
                                                         
            self.scan_progress_percentage = self.counter / len(self.ex_files)
            self.counter += 1
            self.result_json["file_results"].append(file_res)
        except BaseException as be:
          format_exception(be)
          self.result_json["errors"] = 2
          print("Fatal error here!\n")  

        self.status = scanThread.scanStatus.SCAN_FINISHED_SUCCESSFULLY

