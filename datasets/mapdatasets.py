# Basic libs
import os
#import tensorflow as tf
import numpy as np
import time
import glob
import random
import pickle
import copy
import open3d
import torch
from scipy.spatial.distance import cdist
import csv
from ext.benchmark.datasets.datasets import PairwiseDataset, ArgoverseTrackingDataset
# Dataset parent class
#from datasets.common import Dataset
#from datasets.ThreeDMatch import rotate

kitti_icp_cache = {}
kitti_cache = {}
eps = 1e-6


def rotation_matrix(augment_axis, augment_rotation):
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if augment_axis == 1:
        return random.choice([Rx, Ry, Rz])
    return Rx @ Ry @ Rz


def translation_matrix(augment_translation):
    T = np.random.rand(3) * augment_translation
    return T


def make_open3d_point_cloud(xyz, color=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = open3d.utility.Vector3dVector(color)
    return pcd


def read_csv_file(path_file):
    lines = []
    with open(path_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            if i >= 1:
                lines.append(np.asarray(
                    line[0].split(',')[1:]).astype('float'))
    return lines


def make_open3d_feature(data, dim, npts):
    feature = open3d.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.astype('d').transpose()
    return feature


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = open3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(
            point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def rotate(points, num_axis=1):
    if num_axis == 1:
        theta = np.random.rand() * 2 * np.pi
        axis = np.random.randint(3)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]], dtype=np.float32)
        R[:, axis] = 0
        R[axis, :] = 0
        R[axis, axis] = 1
        points = np.matmul(points, R)
    elif num_axis == 3:
        for axis in [0, 1, 2]:
            theta = np.random.rand() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]],
                         dtype=np.float32)
            R[:, axis] = 0
            R[axis, :] = 0
            R[axis, axis] = 1
            points = np.matmul(points, R)
    else:
        exit(-1)
    return points


def pred_to_matrix_np(pred):

    cam_T = np.tile(np.eye(4)[np.newaxis, :, :], [pred.shape[0], 1, 1])
    cam_T[:, 0:3, 3] = pred[:, 0:3]
    s = np.tile(np.linalg.norm(pred[:, 3:], axis=1)[:, np.newaxis], [1, 4])

    q = pred[:, 3:] / (s+eps)

    cam_T[:, 0, 0] = 1 - 2*s[:, 0]*(q[:, 2]**2 + q[:, 3]**2)
    cam_T[:, 0, 1] = 2 * s[:, 0]*(q[:, 1] * q[:, 2] - q[:, 3] * q[:, 0])
    cam_T[:, 0, 2] = 2 * s[:, 0]*(q[:, 1] * q[:, 3] + q[:, 2] * q[:, 0])
    cam_T[:, 1, 0] = 2 * s[:, 0]*(q[:, 1]*q[:, 2] + q[:, 3]*q[:, 0])
    cam_T[:, 1, 1] = 1 - 2*s[:, 0]*(q[:, 1]**2 + q[:, 3]**2)
    cam_T[:, 1, 2] = 2 * s[:, 0]*(q[:, 2] * q[:, 3] - q[:, 1] * q[:, 0])
    cam_T[:, 2, 0] = 2 * s[:, 0]*(q[:, 1] * q[:, 3] - q[:, 2] * q[:, 0])
    cam_T[:, 2, 1] = 2 * s[:, 0]*(q[:, 2] * q[:, 3] + q[:, 1] * q[:, 0])
    cam_T[:, 2, 2] = 1 - 2*s[:, 0]*(q[:, 1]**2 + q[:, 2]**2)

    return cam_T


class KITTIMapDataset(PairwiseDataset):
    AUGMENT = None
    DATA_FILES = {
        'train': 'kitti/train_kitti.txt',  # log ids
        'val': 'kitti/val_kitti.txt',
        'test': 'kitti/test_kitti.txt'
    }
    TEST_RANDOM_ROTATION = False
    IS_ODOMETRY = True
    MAX_TIME_DIFF = 3

    def __init__(self,  split, cfg, config_d3feat, root, input_threads=8,
                 first_subsampling_dl=0.30):
        #super().__init__(split, cfg)
        self.network_model = 'descriptor'
        self.num_threads = input_threads
        self.root = root
        self.icp_path = "data"  # 'data/kitti/icp'
        self.voxel_size = first_subsampling_dl
        self.matching_search_voxel_size = first_subsampling_dl * 0.5  # 1.5
        self.split = split
        # Initiate containers
        #self.anc_points = {'train': [], 'val': [], 'test': []}
        #self.files = {'train': [], 'val': [], 'test': []}

        self.config = config_d3feat

        self.path_map_dict = os.path.join(
            root, "kitti_map_files_d3feat_%s.pkl" % self.split)

        # to match the dataset file used in the benchmark
        self.split = split
        self.cfg = cfg

        self.read_data()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):  # split, idx):
        drive = self.data["id_log"][idx]  # self.files[self.split][idx][0]
        #t0, t1 = self.files[self.split][idx][1], self.files[self.split][idx][2]

        # LiDAR is the target
        #fname1 = self._get_velodyne_fn(drive,  self.files[self.split][idx][1])
        #xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
        #xyz1 = xyzr1[:, :3]

        xyz1 = self.load_kitti_scan(idx)
        # map is the source
        xyz0 = self.get_local_map(
            self.data["T_map"][idx], self.data["T_map"][idx], str(drive))

        # .to(self.config.device)# # M2
        trans = torch.Tensor(np.linalg.inv(self.data["T_map"][idx]))

        # pcd0 = make_open3d_point_cloud(xyz0)#.cpu().numpy())
        #pcd1 = make_open3d_point_cloud(xyz1)
        # pcd0 = pcd0.voxel_down_sample(self.voxel_size)# open3d.voxel_down_sample(pcd0, self.voxel_size)
        # pcd1 = pcd1.voxel_down_sample(self.voxel_size)# pen3d.voxel_down_sample(pcd1, self.voxel_size)
        unaligned_anc_points = xyz0  # np.array(pcd0.points)
        unaligned_pos_points = xyz1  # np.array(pcd1.points)

        if True:  # self.split == 'train' or self.split == 'val':

            pcd0 = make_open3d_point_cloud(xyz0)  # .cpu().numpy())
            pcd1 = make_open3d_point_cloud(xyz1)
            matching_search_voxel_size = self.matching_search_voxel_size
            matches = get_matching_indices(
                pcd0, pcd1, trans.cpu().numpy(), matching_search_voxel_size)

        # align the two point cloud into one corredinate system.
        matches = np.array(matches)
        # pcd0.transform(trans)

        pcd0.transform(trans.cpu().numpy())
        src_points = np.array(pcd0.points)  # gt point clouds
        tgt_points = np.array(pcd1.points)
        src_pcd = pcd0
        tgt_pcd = pcd1

        #open3d.io.write_point_cloud("src_pcd.ply", pcd0)
        #open3d.io.write_point_cloud("tgt_pcd.ply", pcd1)

        if len(matches) > self.config.num_node:
            sel_corr = matches[np.random.choice(
                len(matches), self.config.num_node, replace=False)]
        else:
            sel_corr = matches

        if self.split == "test":
            gt_trans = torch.inverse(self.list_T_gt[idx])
        else:
            # data augmentation
            gt_trans = np.eye(4).astype(np.float32)
            R = rotation_matrix(self.config.augment_axis,
                                self.config.augment_rotation)
            T = translation_matrix(self.config.augment_translation)
            gt_trans[0:3, 0:3] = R
            gt_trans[0:3, 3] = T

        tgt_pcd.transform(gt_trans)
        src_points = np.array(src_pcd.points)
        tgt_points = np.array(tgt_pcd.points)
        src_points += np.random.rand(
            src_points.shape[0], 3) * self.config.augment_noise
        tgt_points += np.random.rand(
            tgt_points.shape[0], 3) * self.config.augment_noise

        sel_P_src = src_points[sel_corr[:, 0], :].astype(np.float32)
        sel_P_tgt = tgt_points[sel_corr[:, 1], :].astype(np.float32)
        dist_keypts = cdist(sel_P_src, sel_P_src)

        pts0 = src_points
        pts1 = tgt_points
        feat0 = np.ones_like(pts0[:, :1]).astype(np.float32)
        feat1 = np.ones_like(pts1[:, :1]).astype(np.float32)
        if self.config.self_augment:
            feat0[np.random.choice(pts0.shape[0], int(
                pts0.shape[0] * 0.99), replace=False)] = 0
            feat1[np.random.choice(pts1.shape[0], int(
                pts1.shape[0] * 0.99), replace=False)] = 0

        return pts0, pts1, feat0, feat1, sel_corr, dist_keypts


class KITTIMapDatasetOld(torch.utils.data.Dataset):

    AUGMENT = None
    DATA_FILES = {
        'train': 'kitti/train_kitti.txt',  # log ids
        'val': 'kitti/val_kitti.txt',
        'test': 'kitti/test_kitti.txt'
    }
    TEST_RANDOM_ROTATION = False
    IS_ODOMETRY = True
    MAX_TIME_DIFF = 3

    # , load_test=False):
    def __init__(self, root, split, config, input_threads=8, first_subsampling_dl=0.30):
        torch.utils.data.Dataset.__init__(self)  # , 'KITTIMap')
        self.network_model = 'descriptor'
        self.num_threads = input_threads
        #self.load_test = load_test
        self.root = root
        self.icp_path = "data"  # 'data/kitti/icp'
        self.voxel_size = first_subsampling_dl
        self.matching_search_voxel_size = first_subsampling_dl * 0.5  # 1.5
        self.split = split
        # Initiate containers
        #self.anc_points = {'train': [], 'val': [], 'test': []}
        self.files = {'train': [], 'val': [], 'test': []}

        self.config = config

        # [0] #
        self.path_map_dict = os.path.join(
            root, "kitti_map_files_d3feat_%s.pkl" % self.split)
        self.read_data()

        # self.prepare_kitti_ply()#split=split)

    def read_data(self):
        self.load_map_data()

        num_logs = len(self.cfg.dict_log_ids[self.split])

        self.data = {}
        self.data["id_log"] = []
        self.data["ind_frame"] = []
        self.data["raw_points"] = []
        self.data["T_map"] = []
        self.data["path_raw_points"] = []

        for ind_log, id_log in enumerate(self.cfg.dict_log_ids[self.split]):
            path_frames = glob.glob(os.path.join(
                self.cfg.path_dataset, "sequences", id_log, "velodyne/*"))
            path_frames.sort()
            num_frames = len(path_frames)
            print("Reading %s frames from %s: %d/%d " %
                  (num_frames, self.cfg.path_dataset, ind_log, num_logs))

            data_odometry = pykitti.odometry(self.cfg.path_dataset, id_log)
            path_poses = os.path.join(
                self.cfg.path_cmrdata, self.split, "kitti-%s.csv" % id_log)
            list_gt_poses = read_csv_file(path_poses)

            sample_interval = 10 if self.split == "test" else self.cfg.trainging_set_step

            for ind_frame in range(0, num_frames, sample_interval):

                path_raw_points = os.path.join(
                    self.cfg.path_dataset, "sequences", id_log, "velodyne/%06d.bin" % ind_frame)
                T_map = pred_to_matrix_np(
                    list_gt_poses[ind_frame][np.newaxis, [0, 1, 2, 6, 3, 4, 5]])[0]

                points_local_map = self.get_local_map(T_map, T_map,  id_log)

                if points_local_map.shape[0] < self.cfg.min_num_pts_map:
                    continue

                # raw_points = pykitti.utils.load_velo_scan(path_raw_points)[:,0:3]
                # raw_points = raw_points[np.random.permutation(raw_points.shape[0])]

                self.data["id_log"].append(id_log)
                self.data["ind_frame"].append(ind_frame)
                self.data["T_map"].append(T_map)
                self.data["path_raw_points"].append(path_raw_points)

        self.length = len(self.data["id_log"])

        if self.split == "test":
            self.read_list_T_gt()
            print("Loading test samples...")
            self.list_test_sample = []
            for i in range(self.length):
                T_noise = self.list_T_gt[i]
                self.list_test_sample.append(self.get_sample(i, T_noise))

        print(f"num of samples = {self.length}")

    def __len__(self):
        return self.length

    def read_map_data(self):
        if os.path.exists(self.path_map_dict):
            with open(self.path_map_dict, 'rb') as f:
                self.dict_maps = pickle.load(f)
            return

        subset_names = open(self.DATA_FILES[self.split]).read().split()
        self.dict_maps = {}
        for id_log in subset_names:
            path_map = os.path.join(
                self.root, "kitti_maps_cmr_new", "map-%02d_0.05.ply" % (int(id_log)))
            print("Load map : ", path_map)
            pcd = open3d.io.read_point_cloud(path_map)
            pcd = pcd.voxel_down_sample(self.config.first_subsampling_dl)
            pcd, ind = pcd.remove_radius_outlier(
                nb_points=7, radius=self.config.first_subsampling_dl*2)
            # torch.Tensor(np.asarray(pcd.points))#.to(self.config.device)
            self.dict_maps[id_log] = np.asarray(pcd.points)

        with open(self.path_map_dict, 'wb') as f:
            print('Saving map file to ', self.path_map_dict)
            pickle.dump(self.dict_maps, f)
            print('Saved!')

    def get_local_map(self, T_lidar, drive):  # , force_select_points=False):
        dist = np.sqrt(
            ((self.dict_maps[drive] - T_lidar[0:3, 3])**2).sum(axis=1))
        ind_valid_local = dist < self.config.depth_max

        return self.dict_maps[drive][ind_valid_local]

    def prepare_kitti_ply(self):  # , split='train'):
        max_time_diff = self.MAX_TIME_DIFF
        subset_names = open(self.DATA_FILES[self.split]).read().split()
        self.all_pos = []
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(
                self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)

            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4])
                             for fname in fnames])
            path_poses = os.path.join(
                self.config.path_cmrdata, self.split, "kitti-%02d.csv" % drive_id)
            list_gt_poses = read_csv_file(path_poses)

            for i in range(0, len(inames), 2):  # curr_time in inames:

                T = pred_to_matrix_np(np.asarray(
                    list_gt_poses[i][np.newaxis, [0, 1, 2, 6, 3, 4, 5]]))[0]
                xyz0 = self.get_local_map(T, dirname)

                # use the local map as the source
                T = np.linalg.inv(T)

                if xyz0.shape[0] < self.config.num_min_map_points:
                    continue

                self.files[self.split].append(
                    (drive_id, inames[i]))  # , next_time))
                # .to(self.config.device))
                self.all_pos.append(torch.Tensor(T))

#

        self.num = len(self.files[self.split])

    def get_batch_gen(self):  # , split, config):
        def random_balanced_gen():
            import pdb
            pdb.set_trace()
            # Initiate concatenation lists
            anc_points_list = []
            pos_points_list = []
            anc_keypts_list = []
            pos_keypts_list = []
            backup_anc_points_list = []
            backup_pos_points_list = []
            ti_list = []
            ti_list_pos = []
            batch_n = 0

            gen_indices = np.arange(self.num)

            print(gen_indices)
            # Generator loop
            for p_i in gen_indices:

                if self.split == 'test':
                    aligned_anc_points, aligned_pos_points, anc_points, pos_points, matches, trans, flag = self.__getitem__(
                        p_i)  # split, p_i)
                    if flag == False:
                        continue
                else:
                    aligned_anc_points, aligned_pos_points, anc_points, pos_points, matches, trans, flag = self.__getitem__(
                        p_i)  # split, p_i)
                    if flag == False:
                        continue

                anc_id = str(self.files[split][p_i][0]) + \
                    "@" + str(self.files[self.split][p_i][1])
                pos_id = str(self.files[split][p_i][0]) + \
                    "@" + str(self.files[self.split][p_i][2])
                # the backup_points shoule be in the same coordinate
                backup_anc_points = aligned_anc_points
                backup_pos_points = aligned_pos_points
                if self.split == 'test':
                    anc_keypts = np.array([])
                    pos_keypts = np.array([])
                else:
                    # input to the network should be in different coordinates
                    anc_keypts = matches[:, 0]
                    pos_keypts = matches[:, 1]
                    selected_ind = np.random.choice(
                        range(len(anc_keypts)), self.config.keypts_num, replace=False)
                    anc_keypts = anc_keypts[selected_ind]
                    pos_keypts = pos_keypts[selected_ind]
                    pos_keypts += len(anc_points)

                if self.split == 'train' or self.split == 'val':
                    # data augmentations: noise
                    anc_noise = np.random.rand(
                        anc_points.shape[0], 3) * self.config.augment_noise
                    pos_noise = np.random.rand(
                        pos_points.shape[0], 3) * self.config.augment_noise
                    anc_points += anc_noise
                    pos_points += pos_noise
                    # data augmentations: rotation
                    anc_points = rotate(
                        anc_points, num_axis=self.config.augment_rotation)
                    pos_points = rotate(
                        pos_points, num_axis=self.config.augment_rotation)
                    # data augmentations: scale
                    scale = config.augment_scale_min + \
                        (config.augment_scale_max -
                         config.augment_scale_min) * random.random()
                    anc_points = scale * anc_points
                    pos_points = scale * pos_points
                    # data augmentations: translation
                    anc_points = anc_points + \
                        np.random.uniform(-self.config.augment_shift_range,
                                          self.config.augment_shift_range, 3)
                    pos_points = pos_points + \
                        np.random.uniform(-self.config.augment_shift_range,
                                          self.config.augment_shift_range, 3)

                # Add data to current batch
                anc_points_list += [anc_points]
                anc_keypts_list += [anc_keypts]
                pos_points_list += [pos_points]
                pos_keypts_list += [pos_keypts]
                backup_anc_points_list += [backup_anc_points]
                backup_pos_points_list += [backup_pos_points]
                ti_list += [p_i]
                ti_list_pos += [p_i]

                yield (np.concatenate(anc_points_list + pos_points_list, axis=0),  # anc_points
                       np.concatenate(anc_keypts_list, axis=0),  # anc_keypts
                       np.concatenate(pos_keypts_list, axis=0),
                       np.array(ti_list + ti_list_pos,
                                dtype=np.int32),  # anc_obj_index
                       np.array([tp.shape[0] for tp in anc_points_list] + [tp.shape[0]
                                                                           for tp in pos_points_list]),  # anc_stack_length
                       np.array([anc_id, pos_id]),
                       np.concatenate(backup_anc_points_list + \
                                      backup_pos_points_list, axis=0),
                       np.array(trans)
                       )
                # print("\t Yield ", anc_id, pos_id)
                anc_points_list = []
                pos_points_list = []
                anc_keypts_list = []
                pos_keypts_list = []
                backup_anc_points_list = []
                backup_pos_points_list = []
                ti_list = []
                ti_list_pos = []
                import time
                # time.sleep(0.3)

        # Generator types and shapes
        gen_types = (tf.float32, tf.int32, tf.int32, tf.int32,
                     tf.int32, tf.string, tf.float32, tf.float32)
        gen_shapes = ([None, 3], [None], [None], [None],
                      [None], [None], [None, 3], [4, 4])

        return random_balanced_gen, gen_types, gen_shapes

#
    def __getitem__(self, idx):  # split, idx):
        drive = self.files[self.split][idx][0]
        #t0, t1 = self.files[self.split][idx][1], self.files[self.split][idx][2]

        # LiDAR is the target
        fname1 = self._get_velodyne_fn(drive,  self.files[self.split][idx][1])
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
        xyz1 = xyzr1[:, :3]

        # map is the source

        xyz0 = self.get_local_map(np.linalg.inv(self.all_pos[idx]), str(drive))

#
        trans = self.all_pos[idx]  # M2

        pcd0 = make_open3d_point_cloud(xyz0)  # .cpu().numpy())
        pcd1 = make_open3d_point_cloud(xyz1)
        # open3d.voxel_down_sample(pcd0, self.voxel_size)
        pcd0 = pcd0.voxel_down_sample(self.voxel_size)
        # pen3d.voxel_down_sample(pcd1, self.voxel_size)
        pcd1 = pcd1.voxel_down_sample(self.voxel_size)
        unaligned_anc_points = np.array(pcd0.points)
        unaligned_pos_points = np.array(pcd1.points)

        if self.split == 'train' or self.split == 'val':
            matching_search_voxel_size = self.matching_search_voxel_size
            matches = get_matching_indices(
                pcd0, pcd1, trans.cpu().numpy(), matching_search_voxel_size)

        # align the two point cloud into one corredinate system.
        matches = np.array(matches)
        # pcd0.transform(trans)

        pcd0.transform(trans.cpu().numpy())
        src_points = np.array(pcd0.points)  # gt point clouds
        tgt_points = np.array(pcd1.points)
        src_pcd = pcd0
        tgt_pcd = pcd1

        #import pdb; pdb.set_trace()
        #open3d.io.write_point_cloud("src_pcd.ply" , pcd0)
        #open3d.io.write_point_cloud("tgt_pcd.ply" , pcd1)

        if len(matches) > self.config.num_node:
            sel_corr = matches[np.random.choice(
                len(matches), self.config.num_node, replace=False)]
        else:
            sel_corr = matches

        # data augmentation
        gt_trans = np.eye(4).astype(np.float32)
        R = rotation_matrix(self.config.augment_axis,
                            self.config.augment_rotation)
        T = translation_matrix(self.config.augment_translation)
        gt_trans[0:3, 0:3] = R
        gt_trans[0:3, 3] = T

        tgt_pcd.transform(gt_trans)
        src_points = np.array(src_pcd.points)
        tgt_points = np.array(tgt_pcd.points)
        src_points += np.random.rand(
            src_points.shape[0], 3) * self.config.augment_noise
        tgt_points += np.random.rand(
            tgt_points.shape[0], 3) * self.config.augment_noise

        sel_P_src = src_points[sel_corr[:, 0], :].astype(np.float32)
        sel_P_tgt = tgt_points[sel_corr[:, 1], :].astype(np.float32)
        dist_keypts = cdist(sel_P_src, sel_P_src)

        pts0 = src_points
        pts1 = tgt_points
        feat0 = np.ones_like(pts0[:, :1]).astype(np.float32)
        feat1 = np.ones_like(pts1[:, :1]).astype(np.float32)
        if self.config.self_augment:
            feat0[np.random.choice(pts0.shape[0], int(
                pts0.shape[0] * 0.99), replace=False)] = 0
            feat1[np.random.choice(pts1.shape[0], int(
                pts1.shape[0] * 0.99), replace=False)] = 0

        return pts0, pts1, feat0, feat1, sel_corr, dist_keypts

        # return (anc_points, pos_points, unaligned_anc_points, unaligned_pos_points, matches, trans, True)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -
                          2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in kitti_cache:
                kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return kitti_cache[data_path]
            else:
                return kitti_cache[data_path][indices]
        else:
            data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
            odometry = []
            if indices is None:
                fnames = glob.glob(self.root + '/' + self.date +
                                   '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
                indices = sorted([int(os.path.split(fname)[-1][:-4])
                                  for fname in fnames])

            for index in indices:
                filename = os.path.join(data_path, '%010d%s' % (index, ext))
                if filename not in kitti_cache:
                    kitti_cache[filename] = np.genfromtxt(filename)
                    odometry.append(kitti_cache[filename])

            odometry = np.array(odometry)
            return odometry

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0
        else:
            lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

            R = 6378137  # Earth's radius in metres

            # convert to metres
            lat, lon = np.deg2rad(lat), np.deg2rad(lon)
            mx = R * lon * np.cos(lat)
            my = R * lat

            times = odometry.T[-1]
            return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + \
                '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        else:
            fname = self.root + \
                '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
                    drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)




class ArgoverseMapDataset(ArgoverseTrackingDataset):  # PairwiseDataset from the benchmark coe

    def __init__(self,  split, cfg, config_d3feat, root,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):

        super().__init__(split, cfg, )
        
        self.network_model = 'descriptor'
        self.num_threads = input_threads
        self.root = root
        self.icp_path = "data"  # 'data/kitti/icp'
        self.voxel_size = first_subsampling_dl
        self.matching_search_voxel_size = first_subsampling_dl * 0.5  # 1.5
        self.split = split
        # Initiate containers
        #self.anc_points = {'train': [], 'val': [], 'test': []}
        #self.files = {'train': [], 'val': [], 'test': []}

        self.config = config_d3feat

        self.path_map_dict = os.path.join(
            root, "argo_map_files_d3feat_%s.pkl" % self.split)

        # to match the dataset file used in the benchmark
        self.split = split
        self.cfg = cfg

        self.root = root = os.path.join(cfg.path_dataset, 'dataset')
        self.split = phase
        self.cfg = cfg

        self.path_map_dict = os.path.join(
            root, "argo_map_files_d3feat_%s_fcgf.pkl" % self.split)
        # self.read_map_data()
        # self.prepare_kitti_ply()#split=split)
        self.read_data()

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __len__(self):
        return self.length

    def __getitem__(self, idx):  # split, idx):
        drive = self.data["id_log"][idx]
        #t0, t1 = self.files[self.split][idx][1], self.files[self.split][idx][2]

        # LiDAR is the target

        xyz0 = self.load_argo_scan_from_path(self.data["path_raw_points"][idx])
        # map is the source
        xyz1_global = self.get_local_map(
            self.data["T_map"][idx], self.data["T_map"][idx], str(drive))
        # .to(self.config.device)# # M2
        trans_global = np.linalg.inv(self.data["T_map"][idx])

        matching_search_voxel_size = self.matching_search_voxel_size
        # if self.random_scale and random.random() < 0.95:
        #    scale = self.min_scale + \
        #        (self.max_scale - self.min_scale) * random.random()
        #    matching_search_voxel_size *= scale
        #    xyz0 = scale * xyz0
        #    xyz1 = scale * xyz1
#
        # Voxelization
        # xyz0 = torch.from_numpy(xyz0).float()  # xyz0#torch.from_numpy(xyz0)
        #xyz1_align = torch.from_numpy(xyz1).float()
        xyz1_align = self.apply_transform(xyz1_global, trans_global)
        # Make point clouds using voxelized points
        #pcd0 = make_open3d_point_cloud(xyz0[sel0])
        #pcd1 = make_open3d_point_cloud(xyz1[sel1])

        import copy
        if self.split != "test":

            T0 = sample_random_trans(xyz0, self.randg, np.pi / 4)
            T1 = sample_random_trans(xyz1_align, self.randg, np.pi / 4)
            trans = T1 @ np.linalg.inv(T0)

            xyz0 = self.apply_transform(xyz0, T0)
            xyz1 = self.apply_transform(xyz1_align, T1)
        else:
            trans = self.list_T_gt[idx].numpy()
            xyz1 = self.apply_transform(xyz1_align, trans)

        sel0 = ME.utils.sparse_quantize(
            xyz0 / self.voxel_size, return_index=True)[1]
        sel1 = ME.utils.sparse_quantize(
            xyz1 / self.voxel_size, return_index=True)[1]

        unique_xyz0_th = xyz0[sel0]  # [ind_0]
        unique_xyz1_th = xyz1[sel1]  # [ind_1]

        pcd0 = make_open3d_point_cloud(unique_xyz0_th)
        pcd1 = make_open3d_point_cloud(unique_xyz1_th)

        # Get matches
        matches = get_matching_indices(
            pcd0, pcd1, trans, matching_search_voxel_size)

        # Get features
        feats_train0, feats_train1 = [], []

        npts0 = unique_xyz0_th.shape[0]
        npts1 = unique_xyz1_th.shape[0]

        feats_train0.append(torch.ones((npts0, 1)))
        feats_train1.append(torch.ones((npts1, 1)))

        feats0 = torch.cat(feats_train0, 1)
        feats1 = torch.cat(feats_train1, 1)

        coords0 = np.floor(unique_xyz0_th / self.voxel_size)
        coords1 = np.floor(unique_xyz1_th / self.voxel_size)

        #pcd0_align = make_open3d_point_cloud(unique_xyz0_th)
        # pcd0_align.transform(trans)
#
        #print(np.asarray(pcd0_align.points)[np.asarray(matches)[:, 0]])
#
        #print(unique_xyz1_th[np.asarray(matches)[:, 1]])

        #print("single batch = ")
        # print(coords0.shape)
        # print(coords1.shape)

        if False:#len(matches) < 300:  # idx == 113:#len(matches) <
         # 10:#coords0.shape[0] <
            # 10 or
            # coords1.shape
            # [0]
            # <
            # 10:
            print("num matches = ", len(matches))
            #print("matches shape = ", matches.shape)

            print(coords0)
#
#
            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(coords0)
            o3d.io.write_point_cloud("coords0_before_%d.ply" % idx, pcd_target)
#
            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(coords1)
            o3d.io.write_point_cloud("coords1_before_%d.ply" % idx, pcd_target)
#

            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(unique_xyz1_th)
            o3d.io.write_point_cloud("unique_xyz1_th_%d.ply" % idx, pcd_target)
#

            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(unique_xyz0_th)
            o3d.io.write_point_cloud("unique_xyz0_th_%d.ply" % idx, pcd_target)
#
            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(unique_xyz0_th)
            pcd_target.transform(trans)

            o3d.io.write_point_cloud(
                "unique_xyz0_th_trans_%d.ply" % idx, pcd_target)
#

            import pdb
            pdb.set_trace()
            #pcd_target = o3d.geometry.PointCloud()
            #pcd_target.points = o3d.utility.Vector3dVector(coords0)
            #o3d.io.write_point_cloud("coords0.ply" , pcd_target)
#
            #pcd_target = o3d.geometry.PointCloud()
            #pcd_target.points = o3d.utility.Vector3dVector(coords1)
            #o3d.io.write_point_cloud("coords1.ply" , pcd_target)

        if self.transform:  # add noises to the point clouds
            coords0, feats0 = self.transform(coords0, feats0)
            coords1, feats1 = self.transform(coords1, feats1)

        #print("****get_item", idx)
        # print(unique_xyz0_th.shape, unique_xyz1_th.shape, coords0.shape,
        #      coords1.shape,  feats0.shape, feats1.shape, len(matches),
        #      np.asarray(matches).max(axis=0))
        return (unique_xyz0_th, unique_xyz1_th, coords0,
                coords1, feats0.float(), feats1.float(), matches, trans)


    def __getitem__(self, idx):  # split, idx):
        drive = self.data["id_log"][idx]  # self.files[self.split][idx][0]


        xyz1 =self.load_argo_scan_from_path(self.data["path_raw_points"][idx])
        # map is the source
        xyz0 = self.get_local_map(
            self.data["T_map"][idx], self.data["T_map"][idx], str(drive))

        # .to(self.config.device)# # M2
        trans = torch.Tensor(np.linalg.inv(self.data["T_map"][idx]))
        unaligned_anc_points = xyz0  # np.array(pcd0.points)
        unaligned_pos_points = xyz1  # np.array(pcd1.points)

        if True:  # self.split == 'train' or self.split == 'val':

            pcd0 = make_open3d_point_cloud(xyz0)  # .cpu().numpy())
            pcd1 = make_open3d_point_cloud(xyz1)
            matching_search_voxel_size = self.matching_search_voxel_size
            matches = get_matching_indices(
                pcd0, pcd1, trans.cpu().numpy(), matching_search_voxel_size)

        # align the two point cloud into one corredinate system.
        matches = np.array(matches)
        # pcd0.transform(trans)

        pcd0.transform(trans.cpu().numpy())
        src_points = np.array(pcd0.points)  # gt point clouds
        tgt_points = np.array(pcd1.points)
        src_pcd = pcd0
        tgt_pcd = pcd1

        open3d.io.write_point_cloud("src_pcd.ply", pcd0)
        open3d.io.write_point_cloud("tgt_pcd.ply", pcd1)
        import pdb; pdb.set_trace()
        if len(matches) > self.config.num_node:
            sel_corr = matches[np.random.choice(
                len(matches), self.config.num_node, replace=False)]
        else:
            sel_corr = matches

        if self.split == "test":
            gt_trans = torch.inverse(self.list_T_gt[idx])
        else:
            # data augmentation
            gt_trans = np.eye(4).astype(np.float32)
            R = rotation_matrix(self.config.augment_axis,
                                self.config.augment_rotation)
            T = translation_matrix(self.config.augment_translation)
            gt_trans[0:3, 0:3] = R
            gt_trans[0:3, 3] = T

        tgt_pcd.transform(gt_trans)
        src_points = np.array(src_pcd.points)
        tgt_points = np.array(tgt_pcd.points)
        src_points += np.random.rand(
            src_points.shape[0], 3) * self.config.augment_noise
        tgt_points += np.random.rand(
            tgt_points.shape[0], 3) * self.config.augment_noise

        sel_P_src = src_points[sel_corr[:, 0], :].astype(np.float32)
        sel_P_tgt = tgt_points[sel_corr[:, 1], :].astype(np.float32)
        dist_keypts = cdist(sel_P_src, sel_P_src)

        pts0 = src_points
        pts1 = tgt_points
        feat0 = np.ones_like(pts0[:, :1]).astype(np.float32)
        feat1 = np.ones_like(pts1[:, :1]).astype(np.float32)
        if self.config.self_augment:
            feat0[np.random.choice(pts0.shape[0], int(
                pts0.shape[0] * 0.99), replace=False)] = 0
            feat1[np.random.choice(pts1.shape[0], int(
                pts1.shape[0] * 0.99), replace=False)] = 0

        return pts0, pts1, feat0, feat1, sel_corr, dist_keypts

