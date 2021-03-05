import os
import torch
import numpy as np
import pandas as pd
import functools
from pandas.io.parsers import read_csv
import torch.utils.data as data
from scipy.spatial import distance_matrix, ConvexHull
from sklearn.decomposition import PCA
import sys
import os
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from traceback import format_exc

from external.utils.numpy_pointcloud_utils import sample_and_group, dbscan_cluster_and_group

def get_outliers_and_axis_reduction_pca(np_array, stddev_factor=1.5):
    try:
        points_mat = np.matrix(np_array)
        transposed_mat = points_mat.T
        dimensions_mean = [sum(column) / len(column) for column in transposed_mat.tolist()]

        normed_transposed_mat = np.matrix(np.stack([[a - dimensions_mean[i] for a in column] for i, column in enumerate(transposed_mat.tolist())]))
        #normed_mat = normed_transposed_mat.T
        covariance_matrix = np.cov(normed_transposed_mat)

        # eigen vectors should be orthogonal
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        eigen_values = [(e, i) for i, e in enumerate(eigen_values)]

        # sort eigen values descending, and create a feature vector according
        eigen_values.sort(reverse=True, key=lambda a: a[0])

        # alreay returns a transposed matrix of eigen vectors!!!!
        feature_vec = np.stack([eigen_vectors[:,i] for e, i in eigen_values])

        transformed_data = feature_vec * normed_transposed_mat
        reduced_data = None

        if (len(feature_vec) > 2):
            reduced_data = feature_vec[:2] * normed_transposed_mat
            reduced_data = reduced_data.T

        # just some sanity
        if (int((transformed_data[0]).mean()) != 0):
            raise BaseException("Some error has occurred during computation") 

        # actual noise reduction, look at PC1 which has the highest explained variance,
        # and filter out all points which are bigger than stddev_factor * std
        pc_one = transformed_data.tolist()[0]
        std = transformed_data[0].std()
        noise = [idx  for idx,val in enumerate(pc_one) if np.abs(val) >= std * stddev_factor]

        if (len(noise) > 0):
            print("Noise reduction: %d points dropped due to being %f times higher than std (second PC)" % (len(noise), stddev_factor))

        # drop noise
        transformed_data = np.delete(transformed_data, noise, 1)

        if (len(transformed_data) > 2): 
            pc_two = transformed_data.tolist()[1]
            std = transformed_data[1].std()
            noise = [idx  for idx,val in enumerate(pc_two) if np.abs(val) >= std * stddev_factor]

            if (len(noise) > 0):
                print("Noise reduction: %d points dropped due to being %f times higher than std" % (len(noise), stddev_factor))

            # drop noise
            transformed_data = np.delete(transformed_data, noise, 1)

        #restore mean in reduced original data
        original_data_reduced = np.matrix(feature_vec).I * transformed_data
        restored = np.stack([[a + dimensions_mean[i] for a in column] for i, column in enumerate(original_data_reduced.tolist())])
        return (restored.T, reduced_data)
    except Exception as e:
        print("Error ocurred during noise reduction!")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)
        print(e)

def PolygonSort(corners):
    n = len(corners)
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    cornersWithAngles = []
    for x, y in corners:
        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        cornersWithAngles.append((x, y, an))
    cornersWithAngles.sort(key = lambda tup: tup[2])
    return [(x, y) for (x, y, an) in cornersWithAngles]

def PolygonArea(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

class _DstormDataset(data.Dataset):
    def __init__(self, root, workers=8):
        self.root = root

        filelist = []
        if isinstance(self.root, str) and os.path.isdir(self.root):
            for root, dirs, files in os.walk(self.root, topdown=False):
                for name in files:
                    full_path = os.path.join(root, name)
                    filename, extension = os.path.splitext(full_path)
                    if extension == ".csv":
                        filelist.append(full_path)
        elif isinstance(self.root, str) and os.path.isfile(self.root) and os.path.splitext(self.root)[1] == ".csv":
            filelist.append(self.root)
        elif isinstance(self.root, list):
            for r in self.root:
                if isinstance(r, str) and os.path.isdir(r):
                    for root, dirs, files in os.walk(r, topdown=False):
                        for name in files:
                            full_path = os.path.join(root, name)
                            filename, extension = os.path.splitext(full_path)
                            if extension == ".csv":
                                filelist.append(full_path)
                elif isinstance(r, str) and os.path.isfile(r) and os.path.splitext(r)[1] == ".csv":
                    filelist.append(r)
                else:
                    raise ValueError(f"{r} is not a path to a directory or a 'csv' file")
        else:
            raise ValueError(f"{self.root} is not a path to a directory or a 'csv' file")

        pool = Pool(workers)
        try:
            df_rows = []
            with tqdm(total=len(filelist)) as pbar:
                result = pool.imap_unordered(self.parse_row, filelist)
                for r in result:
                    df_rows.append(r)
                    pbar.update(1)
        finally:
            pool.close()
            pool.terminate()


        self.orig_df = pd.DataFrame(df_rows)
        self.groups_df = self.create_groups_df(self.orig_df)

        if self.groups_df is not None:
            self.data = self.groups_df.to_dict(orient='records')
            self.indexes = self.groups_df.index.values
            self.index_to_row = {i: dpoint for (i, dpoint) in zip(self.indexes, self.data)}
        else:
            self.data = None
            self.indexes = None
            self.index_to_row = None

    def __repr__(self):
        return f"{str(self.__class__)}_{self.root}"

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label)
        """
        raise NotImplementedError(f"{self.__class__.__name__} should implement method __getitem__(self, index)")

    def create_groups_df(self, df):
        return None

    def process_pointcloud_df(self, pc):
        """
        Change to manipulate pointcloud DataFrame
        """
        df_row = {'pointcloud': pc, 'num_of_points': len(pc)}
        return df_row

    def parse_full_path(self, full_path):
        """
        Change to parse additional/other fields from full path
        """
        parsed_full_path = {'filename': os.path.basename(full_path)}
        return parsed_full_path

    def parse_row(self, full_path):
        """
        Row parser -
            args (list): list of al arguments the function is getting.
                            passing arguments this way is necessary for pool.imap_unordered
        """
        row = {'full_path': full_path}

        try:
            parsed_full_path = self.parse_full_path(row['full_path'])

            with open(row['full_path'], 'r') as f:
                processed_pointcloud = self.process_pointcloud_df(read_csv(f))

            row = {**row, **parsed_full_path, **processed_pointcloud}

        except Exception as e:
            row['Exception'] = format_exc()

        return row


class _ColocDstormDataset(_DstormDataset):
    def __init__(self,
                 *args,
                 coloc_distance=50,
                 coloc_neighbors=1,
                 use_z=False,
                 noise_reduce=False,
                 stddev_num=1.5,
                 density_drop_threshold=0.0,
                 **kwargs):

        self.coordinates_vector = ['x', 'y', 'z'] if use_z else ['x', 'y']
        self.max_npoints = 0

        print(self.coordinates_vector)
        
        # Colocalization
        self.coloc_distance = coloc_distance
        self.coloc_neighbors = coloc_neighbors
        self.use_z = use_z
        self.noise_reduce = noise_reduce
        self.stddev_num = stddev_num
        self.density_drop_threshold = density_drop_threshold
        # kwargs.pop("use_z") # If some how you are reading this, my god I didn't find any other way to pass parameters to father constructor lol

        super(_ColocDstormDataset, self).__init__(*args, **kwargs)

    def parse_full_path(self, full_path):
        parsed_full_path = super(_ColocDstormDataset, self).parse_full_path(full_path)

        full_path_lower = full_path.lower()
        is_vamp = 'vamp' in full_path_lower
        is_bassoon = 'bassoon' in full_path_lower
        is_naive = 'naive' in full_path_lower
        is_tbb = 'tbb' in full_path_lower
        is_mem_and_sub = 'mem_and_sub' in full_path_lower
        is_membrane_only = 'membrane_only' in full_path_lower
        is_sub_membrane = 'sub_membrane' in full_path_lower
        is_square = 'square' in full_path_lower

        if is_vamp + is_bassoon != 1:
            parsed_full_path['coprotein'] = 'Unknown'
        else:
            if is_bassoon:
                parsed_full_path['coprotein'] = 'Bassoon'
            else:
                parsed_full_path['coprotein'] = 'Vamp'

        if is_naive + is_tbb != 1:
            parsed_full_path['label'] = 'Unknown'
        else:
            if is_naive:
                parsed_full_path['label'] = 'Naive'
            else:
                parsed_full_path['label'] = 'TBB'

        if is_mem_and_sub + is_membrane_only + is_sub_membrane + is_square != 1:
            parsed_full_path['zone'] = 'Unknown'
        else:
            if is_mem_and_sub:
                parsed_full_path['zone'] = 'Mem_and_Sub'
            elif is_membrane_only:
                parsed_full_path['zone'] = 'Membrane_Only'
            elif is_sub_membrane:
                parsed_full_path['zone'] = 'Sub_Membrane'
            else:
                parsed_full_path['zone'] = 'Square'

        return parsed_full_path

    def create_groups_df(self, df):
        #max_npoints = max(df.get('probe0_max_npoints', np.array(0)).max(), df.get('probe1_max_npoints', np.array(0)).max())
        #assert max_npoints > 0, "No grouping have been done to create groups dataframe"
        #self.max_npoints = int(2 ** np.ceil(np.log2(max_npoints)))

        groups_df = []
        for ind, series in df.iterrows():
            base_df = self.parse_full_path(series['full_path'])
            base_df['full_path'] = series['full_path']
            base_df = pd.DataFrame([base_df])            

            if series.get('probe0_ngroups', 0) > 0:
                base_df['probe'] = 0                
                base_df_pad = pd.concat(len(series.probe0_groups_df) * [base_df], ignore_index=True)                
                groups_df.append(pd.concat([base_df_pad, series['probe0_groups_df']], axis=1))

            if series.get('probe1_ngroups', 0) > 0:
                base_df['probe'] = 1
                base_df_pad = pd.concat(len(series.probe1_groups_df) * [base_df], ignore_index=True)
                groups_df.append(pd.concat([base_df_pad, series['probe1_groups_df']], axis=1))

        if len(groups_df) >= 1:
            return pd.concat(groups_df, ignore_index=True)

        return None;

    def grouping_function(self, points):
        raise NotImplementedError()

    def find_groups(self, pc):
        points = pc[self.coordinates_vector].to_numpy()
        if points.shape[1] == 2:
            points = np.concatenate([points, np.zeros((len(points), 1), dtype=points.dtype)], axis=1)
        print("\n\n#####################################\n\n")
        print(points)
        print("\n\n#####################################\n\n")
        print("Calculating xy plane convex hull, PCA")
        print("Noise reducing with PCA" if self.noise_reduce else "No noise reduction")
        centroids, groups, unassigned = self.grouping_function(points)

        if len(groups) > 0:
            max_npoints = max([len(g) for g in groups])
        else:
            max_npoints = 0

        unassigned = pc.iloc[unassigned].copy()

        groups_df_rows = []
        for centroid, group in zip(centroids, groups):
            groups_df_row = {}
            groups_df_row['centroid'] = centroid
            groups_df_row['group'] = group

            try:
                pointcloud = pc.iloc[group].copy()
                groups_df_row['pointcloud'] = pointcloud

                pca_pc = pointcloud[self.coordinates_vector].to_numpy()
                nr, reduced_cluster = get_outliers_and_axis_reduction_pca(pca_pc, stddev_factor=self.stddev_num)

                if (self.noise_reduce):                    
                    groups_df_row['noise_reduced_clusters'] = nr
                    pca_pc = nr
                    xy_plane_pc = nr[:,[0,1]]
                else: 
                    groups_df_row['noise_reduced_clusters'] = []
                    xy_plane_pc = pointcloud[["x","y"]].to_numpy()
                
                groups_df_row['num_of_points'] = pca_pc.shape[0]

                convex_hull = ConvexHull(xy_plane_pc)
                groups_df_row['convex_hull'] = xy_plane_pc[convex_hull.simplices]
                corners = list(set(functools.reduce(lambda x,y: x+y, [[(a,b) for a,b in x] for x in xy_plane_pc[convex_hull.simplices]])))
                groups_df_row['polygon_size'] = PolygonArea(PolygonSort(corners))
                groups_df_row['polygon_density'] = float((groups_df_row['num_of_points'] * 10)) / groups_df_row['polygon_size']

                if (self.density_drop_threshold > 0.0):
                    if (groups_df_row['polygon_density'] < self.density_drop_threshold):
                        print("Dropping cluster due to density (%f < %f)" % (groups_df_row['polygon_density'], self.density_drop_threshold))
                        unassigned = pd.concat([unassigned, groups_df_row['pointcloud']])
                        continue

                if reduced_cluster is not None:
                    reduced_convex_hull = ConvexHull(reduced_cluster)
                    corners = list(set(functools.reduce(lambda x,y: x+y,[[(a.tolist()[0][0], a.tolist()[0][1]) for a in x] for x in reduced_cluster[reduced_convex_hull.simplices]])))
                    groups_df_row['reduced_polygon_size'] = PolygonArea(PolygonSort(corners))
                    groups_df_row['reduced_polygon_density'] = float((groups_df_row['num_of_points'] * 10)) / groups_df_row['reduced_polygon_size']
                else:
                    groups_df_row['reduced_polygon_size'] = None
                    groups_df_row['reduced_polygon_density'] = None


                pca = PCA()
                pca.fit(pca_pc)
                groups_df_row['pca_components'] = pca.components_
                groups_df_row['pca_mean'] = pca.mean_
                groups_df_row['pca_std'] = np.sqrt(pca.explained_variance_)
                groups_df_row['pca_size'] = np.sqrt(np.prod(groups_df_row['pca_std']))

            except Exception as e:
                print("Error ocurred during analysis of pc:")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e, exc_type, fname, exc_tb.tb_lineno)
                groups_df_row['Exception'] = format_exc()

            groups_df_rows.append(groups_df_row)

        return (pd.DataFrame(groups_df_rows), len(groups_df_rows), max_npoints, unassigned)

    @staticmethod
    def colocalization(df0, df1, min_distance=50, min_neighbors=1):
        v0 = df0[['x', 'y']].to_numpy()
        v1 = df1[['x', 'y']].to_numpy()

        dist = distance_matrix(v0, v1, p=2)
        mask = np.zeros_like(dist)
        mask[dist < min_distance] = 1

        coloc0 = mask.sum(axis=1)  # Sum all columns to get rows colocalizations
        coloc0[coloc0 < min_neighbors] = 0

        coloc1 = mask.sum(axis=0)  # Sum all rows to get columns colocalizations
        coloc1[coloc1 < min_neighbors] = 0

        return coloc0, coloc1

    @staticmethod
    def colocalization_ratio(colocalization_vector, colocalization_total):
        return 100.0 * colocalization_vector.sum() / colocalization_total

    def process_pointcloud_df(self, pc):
        df_row = super(_ColocDstormDataset, self).process_pointcloud_df(pc)
        try:
            pc_probe0 = pc.query('probe == 0')
            pc_probe1 = pc.query('probe == 1')

            df_row['probe0_num_of_points'] = len(pc_probe0.index)
            df_row['probe1_num_of_points'] = len(pc_probe1.index)

            ### Grouping ###
            if df_row['probe0_num_of_points'] > 0:
                df_row['probe0_groups_df'], df_row['probe0_ngroups'], df_row['probe0_max_npoints'], df_row['probe_0_unassigned'] = \
                    self.find_groups(pc_probe0)

            if df_row['probe1_num_of_points'] > 0:
                df_row['probe1_groups_df'], df_row['probe1_ngroups'], df_row['probe1_max_npoints'], df_row['probe_1_unassigned'] = \
                    self.find_groups(pc_probe1)

            ### Colocalization analysis ###
            if df_row['probe0_num_of_points'] == 0 or df_row['probe1_num_of_points'] == 0:
                df_row['colocalization_available'] = False
            else:
                df_row['colocalization_available'] = True
                coloc0, coloc1 = self.colocalization(pc_probe0, pc_probe1, self.coloc_distance, self.coloc_neighbors)

                df_row['probe0_colocalize'] = self.colocalization_ratio((coloc0 > 0), coloc0.shape[0])
                df_row['probe1_colocalize'] = self.colocalization_ratio((coloc1 > 0), coloc1.shape[0])

                pc.loc[pc_probe0.index.values, 'colocalization'] = coloc0
                pc.loc[pc_probe1.index.values, 'colocalization'] = coloc1

        except Exception as e:
            df_row['Exception'] = format_exc()

        return df_row


class DstormDatasetSimpleSphereCluster(_ColocDstormDataset):
    def __init__(self, *args, radius=500, max_nsamples=80, min_npoints=128, **kwargs):
        # Sampling
        self.radius = radius
        self.max_nsamples = max_nsamples
        self.min_npoints = min_npoints

        super(DstormDatasetSimpleSphereCluster, self).__init__(*args, **kwargs)

    def grouping_function(self, points):
        return sample_and_group(points, self.radius, self.max_nsamples, self.min_npoints)


class DstormDatasetDBSCAN(_ColocDstormDataset):
    def __init__(self, 
                 *args, 
                 dbscan_eps=15, 
                 dbscan_min_samples=5, 
                 min_npoints=15, 
                 max_std_distance=2.5,
                 use_hdbscan=False,
                 hdbscan_min_npoints=15,
                 hdbscan_min_samples=1,
                 hdbscan_epsilon_threshold=-9999,
                 hdbscan_extracting_alg="leaf",
                 hdbscan_alpha=1.0,
                 **kwargs):

        # Sampling
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_npoints = min_npoints
        self.max_std_distance = max_std_distance
        self.use_hdbscan = use_hdbscan
        self.hdbscan_min_npoints = hdbscan_min_npoints
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_epsilon_threshold = hdbscan_epsilon_threshold
        self.hdbscan_extracting_alg = hdbscan_extracting_alg
        self.hdbscan_alpha = hdbscan_alpha

        super(DstormDatasetDBSCAN, self).__init__(*args, **kwargs)

    def grouping_function(self, points):
        centroids, groups, unassigned = dbscan_cluster_and_group(
            points,
            eps=self.dbscan_eps,
            min_npoints=self.dbscan_min_samples,
            min_cluster_points=self.min_npoints,
            max_std_distance=self.max_std_distance,
            use_hdbscan=self.use_hdbscan,
            hdbscan_min_cluster_points=self.hdbscan_min_npoints,
            hdbscan_min_samples=self.hdbscan_min_samples,
            hdbscan_epsilon_threshold=self.hdbscan_epsilon_threshold,
            hdbscan_extracting_alg=self.hdbscan_extracting_alg,
            hdbscan_alpha=self.hdbscan_alpha)
        
        return centroids, groups, unassigned

#
# class _NoasDstormDataset3D(_NoasDstormDataset):
#     def find_groups(self, df):
#         xyz = df[['x', 'y', 'z']].to_numpy()
#         centroids, groups = sample_and_group(xyz, self.radius, self.max_nsamples, self.min_npoints)
#         if len(groups) > 0:
#             max_npoints = max([len(g) for g in groups])
#         else:
#             max_npoints = 0
#         return groups, centroids, len(groups), max_npoints
#
#     @staticmethod
#     def colocalization(df0, df1, min_distance=50, min_neighbors=1):
#         v0 = df0[['x', 'y', 'z']].to_numpy()
#         v1 = df1[['x', 'y', 'z']].to_numpy()
#
#         dist = distance_matrix(v0, v1, p=2)
#         l0, l1 = dist.shape
#
#         mask = np.zeros_like(dist)
#         mask[dist < min_distance] = 1
#
#         coloc0 = mask.sum(axis=1)  # Sum all columns to get rows colocalizations
#         coloc0[coloc0 < min_neighbors] = 0
#
#         coloc1 = mask.sum(axis=0)  # Sum all rows to get columns colocalizations
#         coloc1[coloc1 < min_neighbors] = 0
#
#         return coloc0, coloc1
#
#
# class NoasDstormDatasetSimpleXY(_NoasDstormDataset):
#     labels_dict = {
#         'Naive': 0,
#         'TBB': 1
#     }
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (pointcloud, target, index) where target is class_index of the target class.
#         """
#         row = self.index_to_row[index]
#
#         xy = row['pointcloud'][['x', 'y']].to_numpy()
#         xyz = np.concatenate([xy, np.zeros((len(xy), 1), dtype=xy.dtype)], axis=1)
#         xyz = np.pad(xyz, ((0, self.max_npoints - len(xyz)), (0, 0)), constant_values=np.nan)
#
#         # Normalize
#         xyz = xyz - row['centroid']
#         xyz = xyz / self.radius
#
#         pointcloud = torch.from_numpy(xyz).type(dtype=torch.float)
#         label = torch.tensor(self.labels_dict[row["label"]], dtype=torch.long)
#
#         return pointcloud, label, index


# if __name__ == '__main__':
#     from datasets.samplers import NoasGroupSampler
#     from torch.utils.data import DataLoader

#     batch_size = 10

#     dataset = DstormDatasetDBSCAN(
#         root='/home/oronlevy/ext/data/playground/',
#         dbscan_eps=100,
#         dbscan_min_samples=16,
#         min_npoints=0,
#         max_std_distance=2.5,
#         coloc_distance=50,
#         coloc_neighbors=1,
#         workers=1
#     )
#     train_sampler, valid_sampler = NoasGroupSampler(dataset, ratio=0.8)
#     trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
#                              sampler=train_sampler, pin_memory=False)

#     print(len(dataset))
#     print(len(train_sampler), len(valid_sampler))
#     print(len(trainloader))

#     i = trainloader.__iter__()
#     data = i.__next__()
#     pass
