import os
import torch
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
import torch.utils.data as data
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA

from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from traceback import format_exc

from utils.numpy_pointcloud_utils import sample_and_group, dbscan_cluster_and_group


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

        self.data = self.groups_df.to_dict(orient='records')
        self.indexes = self.groups_df.index.values
        self.index_to_row = {i: dpoint for (i, dpoint) in zip(self.indexes, self.data)}

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
    def __init__(self, *args, coloc_distance=50, coloc_neighbors=1, **kwargs):
        self.coordinates_vector = ['x', 'y']
        self.max_npoints = 0

        # Colocalization
        self.coloc_distance = coloc_distance
        self.coloc_neighbors = coloc_neighbors

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
        max_npoints = max(df.get('probe0_max_npoints', np.array(0)).max(), df.get('probe1_max_npoints', np.array(0)).max())
        assert max_npoints > 0, "No grouping have been done to create groups dataframe"
        self.max_npoints = int(2 ** np.ceil(np.log2(max_npoints)))

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

        return pd.concat(groups_df, ignore_index=True)

    def grouping_function(self, points):
        raise NotImplementedError()

    def find_groups(self, pc):
        points = pc[self.coordinates_vector].to_numpy()
        if points.shape[1] == 2:
            points = np.concatenate([points, np.zeros((len(points), 1), dtype=points.dtype)], axis=1)
        centroids, groups = self.grouping_function(points)

        if len(groups) > 0:
            max_npoints = max([len(g) for g in groups])
        else:
            max_npoints = 0

        groups_df_rows = []
        for centroid, group in zip(centroids, groups):
            groups_df_row = {}
            groups_df_row['centroid'] = centroid
            groups_df_row['group'] = group
            groups_df_row['num_of_points'] = len(group)

            try:
                pointcloud = pc.iloc[group].copy()
                groups_df_row['pointcloud'] = pointcloud

                pca = PCA()
                pca.fit(pointcloud[self.coordinates_vector].to_numpy())
                groups_df_row['pca_components'] = pca.components_
                groups_df_row['pca_mean'] = pca.mean_
                groups_df_row['pca_std'] = np.sqrt(pca.explained_variance_)
                groups_df_row['pca_size'] = np.sqrt(np.prod(groups_df_row['pca_std']))

            except Exception as e:
                groups_df_row['Exception'] = format_exc()

            groups_df_rows.append(groups_df_row)

        return pd.DataFrame(groups_df_rows), len(groups), max_npoints

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
                df_row['probe0_groups_df'], df_row['probe0_ngroups'], df_row['probe0_max_npoints'] = \
                    self.find_groups(pc_probe0)

            if df_row['probe1_num_of_points'] > 0:
                df_row['probe1_groups_df'], df_row['probe1_ngroups'], df_row['probe1_max_npoints'] = \
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
    def __init__(self, *args, dbscan_eps=15, dbscan_min_samples=5, min_npoints=15, max_std_distance=2.5, **kwargs):
        # Sampling
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_npoints = min_npoints
        self.max_std_distance = max_std_distance

        super(DstormDatasetDBSCAN, self).__init__(*args, **kwargs)

    def grouping_function(self, points):
        print("Are we here!?!?!?!\n")
        centroids, groups = dbscan_cluster_and_group(
            points,
            eps=self.dbscan_eps,
            min_npoints=self.dbscan_min_samples,
            min_cluster_points=self.min_npoints,
            max_std_distance=self.max_std_distance
        )
        return centroids, groups

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


if __name__ == '__main__':
    from datasets.samplers import NoasGroupSampler
    from torch.utils.data import DataLoader

    batch_size = 10

    dataset = DstormDatasetDBSCAN(
        root='/home/oronlevy/ext/data/playground/',
        dbscan_eps=100,
        dbscan_min_samples=16,
        min_npoints=0,
        max_std_distance=2.5,
        coloc_distance=50,
        coloc_neighbors=1,
        workers=1
    )
    train_sampler, valid_sampler = NoasGroupSampler(dataset, ratio=0.8)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                             sampler=train_sampler, pin_memory=False)

    print(len(dataset))
    print(len(train_sampler), len(valid_sampler))
    print(len(trainloader))

    i = trainloader.__iter__()
    data = i.__next__()
    pass
