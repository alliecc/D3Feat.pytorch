import os
import open3d as o3d
import argparse
import json
import importlib
import logging
import torch
import numpy as np
from multiprocessing import Process, Manager
from functools import partial
from easydict import EasyDict as edict
from utils.pointcloud import make_point_cloud
from models.architectures import KPFCNN
from utils.timer import Timer, AverageMeter
from datasets.ThreeDMatch import ThreeDMatchTestset
from datasets.dataloader import get_dataloader
from datasets.mapdatasets import KITTIMapDataset
from geometric_registration.common import get_pcd, get_keypts, get_desc, get_scores, loadlog, build_correspondence


def register_one_scene(inlier_ratio_threshold, distance_threshold, save_path, return_dict, scene):
    gt_matches = 0
    pred_matches = 0
    keyptspath = f"{save_path}/keypoints/{scene}"
    descpath = f"{save_path}/descriptors/{scene}"
    scorepath = f"{save_path}/scores/{scene}"
    gtpath = f'geometric_registration/gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)
    inlier_num_meter, inlier_ratio_meter = AverageMeter(), AverageMeter()
    pcdpath = f"{config.root}/fragments/{scene}/"
    num_frag = len([filename for filename in os.listdir(
        pcdpath) if filename.endswith('ply')])
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            cloud_bin_s = f'cloud_bin_{id1}'
            cloud_bin_t = f'cloud_bin_{id2}'
            key = f"{id1}_{id2}"
            if key not in gtLog.keys():
                # skip the pairs that have less than 30% overlap.
                num_inliers = 0
                inlier_ratio = 0
                gt_flag = 0
            else:
                source_keypts = get_keypts(keyptspath, cloud_bin_s)
                target_keypts = get_keypts(keyptspath, cloud_bin_t)
                source_desc = get_desc(descpath, cloud_bin_s, 'D3Feat')
                target_desc = get_desc(descpath, cloud_bin_t, 'D3Feat')
                source_score = get_scores(
                    scorepath, cloud_bin_s, 'D3Feat').squeeze()
                target_score = get_scores(
                    scorepath, cloud_bin_t, 'D3Feat').squeeze()
                source_desc = np.nan_to_num(source_desc)
                target_desc = np.nan_to_num(target_desc)

                # randomly select 5000 keypts
                if args.random_points:
                    source_indices = np.random.choice(
                        range(source_keypts.shape[0]), args.num_points)
                    target_indices = np.random.choice(
                        range(target_keypts.shape[0]), args.num_points)
                else:
                    source_indices = np.argsort(
                        source_score)[-args.num_points:]
                    target_indices = np.argsort(
                        target_score)[-args.num_points:]
                source_keypts = source_keypts[source_indices, :]
                source_desc = source_desc[source_indices, :]
                target_keypts = target_keypts[target_indices, :]
                target_desc = target_desc[target_indices, :]

                corr = build_correspondence(source_desc, target_desc)

                gt_trans = gtLog[key]
                frag1 = source_keypts[corr[:, 0]]
                frag2_pc = o3d.geometry.PointCloud()
                frag2_pc.points = o3d.utility.Vector3dVector(
                    target_keypts[corr[:, 1]])
                frag2_pc.transform(gt_trans)
                frag2 = np.asarray(frag2_pc.points)
                distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
                num_inliers = np.sum(distance < distance_threshold)
                inlier_ratio = num_inliers / len(distance)
                if inlier_ratio > inlier_ratio_threshold:
                    pred_matches += 1
                gt_matches += 1
                inlier_num_meter.update(num_inliers)
                inlier_ratio_meter.update(inlier_ratio)
    recall = pred_matches * 100.0 / gt_matches
    return_dict[scene] = [recall, inlier_num_meter.avg, inlier_ratio_meter.avg]
    logging.info(f"{scene}: Recall={recall:.2f}%, inlier ratio={inlier_ratio_meter.avg*100:.2f}%, inlier num={inlier_num_meter.avg:.2f}")
    return recall, inlier_num_meter.avg, inlier_ratio_meter.avg


def generate_features(model, dloader, config, chosen_snapshot):
    dataloader_iter = dloader.__iter__()

    descriptor_path = f'{save_path}/descriptors'
    keypoint_path = f'{save_path}/keypoints'
    score_path = f'{save_path}/scores'
    if not os.path.exists(descriptor_path):
        os.mkdir(descriptor_path)
    if not os.path.exists(keypoint_path):
        os.mkdir(keypoint_path)
    if not os.path.exists(score_path):
        os.mkdir(score_path)

    # generate descriptors
    recall_list = []

    list_results_to_save = []
    for i, inputs in enumerate(dloader):

        # for scene in dset.scene_list:
        #    descriptor_path_scene = os.path.join(descriptor_path, scene)
        #    keypoint_path_scene = os.path.join(keypoint_path, scene)
        #    score_path_scene = os.path.join(score_path, scene)
        #    if not os.path.exists(descriptor_path_scene):
        #        os.mkdir(descriptor_path_scene)
        #    if not os.path.exists(keypoint_path_scene):
        #        os.mkdir(keypoint_path_scene)
        #    if not os.path.exists(score_path_scene):
        #        os.mkdir(score_path_scene)
        #    pcdpath = f"{config.root}/fragments/{scene}/"
        #    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
        #    # generate descriptors for each fragment
        # for ids in range(num_frag):
        #inputs = dataloader_iter.next()

        for k, v in inputs.items():  # load inputs to device.
            if type(v) == list:
                inputs[k] = [item.cuda() for item in v]
            else:
                inputs[k] = v.cuda()
        features, scores = model(inputs)

        first_pcd_length, second_pcd_length = inputs['stack_lengths'][0]

        pts0, feat0, scores0 = inputs['points'][0][:int(first_pcd_length)], features[:int(
            first_pcd_length)], scores[:int(first_pcd_length)]
        pts1, feat1, scores1 = inputs['points'][0][int(first_pcd_length):], features[int(
            first_pcd_length):], scores[int(first_pcd_length):]
        print(i)
        dict_sample = {"pts_source": pts1.cpu().detach().numpy(),
                       "feat_source": feat1.cpu().detach().numpy(),
                       "score_source": scores1.cpu().detach().numpy(),
                       "pts_target": pts0.cpu().detach().numpy(),
                       "feat_target": feat0.cpu().detach().numpy(),
                       "score_target": scores0.cpu().detach().numpy()}

        list_results_to_save.append(dict_sample)
        #from ext.benchmark.utils.utils import save_list_pc
        #save_list_pc(["0.ply", "1.ply"], [pts0.cpu().detach().numpy(), pts1.cpu().detach().numpy()], [feat0.cpu().detach().numpy(), feat1.cpu().detach().numpy()])
        #save_list_pc(["0s.ply", "1s.ply"], [pts0.cpu().detach().numpy(), pts1.cpu().detach().numpy()], [scores0.cpu().detach().numpy(), scores1.cpu().detach().numpy()])
        #import pdb; pdb.set_trace()
        #pts0 = inputs['points'][0][:int(first_pcd_length)]
#
        #pcd_size = inputs['stack_lengths'][0][0]
        #pts = inputs['points'][0][:int(pcd_size)]
        #features, scores = features[:int(pcd_size)], scores[:int(pcd_size)]
#
        ## scores = torch.ones_like(features[:, 0:1])
#
#
        # np.save(f'{descriptor_path_scene}/cloud_bin_{ids}.D3Feat', features.detach().cpu().numpy().astype(np.float32))
        # np.save(f'{keypoint_path_scene}/cloud_bin_{ids}', pts.detach().cpu().numpy().astype(np.float32))
        # np.save(f'{score_path_scene}/cloud_bin_{ids}', scores.detach().cpu().numpy().astype(np.float32))
        # print(f"Generate cloud_bin_{ids} for {scene}")

    import pickle
    path_results_to_save = "d3feat.results.pkl"
    print('Saving results to ', path_results_to_save)
    pickle.dump(list_results_to_save, open(path_results_to_save, 'wb'))
    print('Saved!')
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='',
                        type=str, help='snapshot dir')
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--distance_threshold', default=0.3, type=float)
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=250, type=int)
    parser.add_argument('--generate_features',
                        default=False, action='store_true')
    args = parser.parse_args()
    # if args.random_points:
    #    log_filename = f'geometric_registration/{args.chosen_snapshot}-rand-{args.num_points}.log'
    # else:
    #    log_filename = f'geometric_registration/{args.chosen_snapshot}-pred-{args.num_points}.log'
    # logging.basicConfig(level=logging.INFO,
    #    filename=log_filename,
    #    filemode='w',
    #    format="")
#

    config_path = f'./data/D3Feat/snapshot/{args.chosen_snapshot}/config.json'
    config_default = json.load(open(config_path, 'r'))
    config_default = edict(config_default)

    from training_kitti_map import KITTIConfig, ConfigObj
    kitti_config = KITTIConfig()
    config = {}
    config.update(vars(args))
    config.update(config_default)
    config.update(kitti_config.__dict__)
    config = ConfigObj(config)

    # create model
    # config.architecture = [
    #    'simple',
    #    'resnetb',
    # ]
    # for i in range(config.num_layers-1):
    #    config.architecture.append('resnetb_strided')
    #    config.architecture.append('resnetb')
    #    config.architecture.append('resnetb')
    # for i in range(config.num_layers-2):
    #    config.architecture.append('nearest_upsample')
    #    config.architecture.append('unary')
    # config.architecture.append('nearest_upsample')
    # config.architecture.append('last_unary')
#
    # # dynamically load the model from snapshot
    # module_file_path = f'snapshot/{chosen_snap}/model.py'
    # module_name = 'model'
    # module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    # module = importlib.util.module_from_spec(module_spec)
    # module_spec.loader.exec_module(module)
    # model = module.KPFCNN(config)

    # if test on datasets with different scale
    # config.first_subsampling_dl = [new voxel size for first layer]

    model = KPFCNN(config)

    model.load_state_dict(torch.load(f'./data/D3Feat/snapshot/{args.chosen_snapshot}/models/model_best_acc.pth')['state_dict'])
    print(f"Load weight from snapshot/{args.chosen_snapshot}/models/model_best_acc.pth")
    model.eval()

    save_path = f'geometric_registration/{args.chosen_snapshot}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.generate_features:
        import importlib
        cfg = importlib.import_module("configs.config")

        dset = KITTIMapDataset(
            "test", cfg, config_d3feat=config, root=config.root)
        # dset = ThreeDMatchTestset(root=config.root,
        #                    downsample=config.downsample,
        #                    config=config,
        #                    last_scene=False,
        #                )
        dloader, _ = get_dataloader(dataset=dset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    )
        generate_features(model.cuda(), dloader, config, args.chosen_snapshot)

    def test_kitti(model, dataset, config):
        # self.sess.run(dataset.test_init_op)
        import sys
        use_random_points = False
        icp_save_path = "d3feat_output"
        if use_random_points:
            num_keypts = 5000
            # icp_save_path = f'geometric_registration_kitti/D3Feat_{self.experiment_str}-rand{num_keypts}'
        else:
            num_keypts = 250
            # icp_save_path = f'geometric_registration_kitti/D3Feat_{self.experiment_str}-pred{num_keypts}'
        if not os.path.exists(icp_save_path):
            os.mkdir(icp_save_path)

        ch = logging.StreamHandler(sys.stdout)
        logging.getLogger().setLevel(logging.INFO)
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d %H:%M:%S', handlers=[ch])

        success_meter, loss_meter, rte_meter, rre_meter = AverageMeter(
        ), AverageMeter(), AverageMeter(), AverageMeter()
        feat_timer, reg_timer = Timer(), Timer()

        for i in range(dataset.length):
            import pdb
            pdb.set_trace()
            # feat_timer.tic()
            ops = [model.anchor_inputs, model.out_features,
                   model.out_scores, model.anc_id, model.pos_id, model.accuracy]
            [inputs, features, scores, anc_id, pos_id, accuracy] = self.sess.run(
                ops, {model.dropout_prob: 1.0})
            # feat_timer.toc()
            # print(accuracy, anc_id)

            stack_lengths = inputs['stack_lengths']
            first_pcd_indices = np.arange(stack_lengths[0])
            second_pcd_indices = np.arange(stack_lengths[1]) + stack_lengths[0]
            # anc_points = inputs['points'][0][first_pcd_indices]
            # pos_points = inputs['points'][0][second_pcd_indices]
            # anc_features = features[first_pcd_indices]
            # pos_features = features[second_pcd_indices]
            # anc_scores = scores[first_pcd_indices]
            # pos_scores = scores[second_pcd_indices]
            if use_random_points:
                anc_keypoints_id = np.random.choice(
                    stack_lengths[0], num_keypts)
                pos_keypoints_id = np.random.choice(
                    stack_lengths[1], num_keypts) + stack_lengths[0]
                anc_points = inputs['points'][0][anc_keypoints_id]
                pos_points = inputs['points'][0][pos_keypoints_id]
                anc_features = features[anc_keypoints_id]
                pos_features = features[pos_keypoints_id]
                anc_scores = scores[anc_keypoints_id]
                pos_scores = scores[pos_keypoints_id]
            else:
                scores_anc_pcd = scores[first_pcd_indices]
                scores_pos_pcd = scores[second_pcd_indices]
                anc_keypoints_id = np.argsort(
                    scores_anc_pcd, axis=0)[-num_keypts:].squeeze()
                pos_keypoints_id = np.argsort(
                    scores_pos_pcd, axis=0)[-num_keypts:].squeeze() + stack_lengths[0]
                anc_points = inputs['points'][0][anc_keypoints_id]
                anc_features = features[anc_keypoints_id]
                anc_scores = scores[anc_keypoints_id]
                pos_points = inputs['points'][0][pos_keypoints_id]
                pos_features = features[pos_keypoints_id]
                pos_scores = scores[pos_keypoints_id]

            pcd0 = make_open3d_point_cloud(anc_points)
            pcd1 = make_open3d_point_cloud(pos_points)
            feat0 = make_open3d_feature(
                anc_features, 32, anc_features.shape[0])
            feat1 = make_open3d_feature(
                pos_features, 32, pos_features.shape[0])

            reg_timer.tic()
            filename = anc_id.decode(
                "utf-8") + "-" + pos_id.decode("utf-8").split("@")[-1] + '.npz'
            if os.path.exists(join(icp_save_path, filename)):
                data = np.load(join(icp_save_path, filename))
                T_ransac = data['trans']
                print(f"Read from {join(icp_save_path, filename)}")
            else:

                distance_threshold = dataset.voxel_size * 1.0
                ransac_result = open3d.registration.registration_ransac_based_on_feature_matching(
                    pcd0, pcd1, feat0, feat1, distance_threshold,
                    open3d.registration.TransformationEstimationPointToPoint(False), 4, [
                        open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(
                            0.9),
                        open3d.registration.CorrespondenceCheckerBasedOnDistance(
                            distance_threshold)
                    ],
                    open3d.registration.RANSACConvergenceCriteria(50000, 1000)
                    # open3d.registration.RANSACConvergenceCriteria(4000000, 10000)
                )
                # print(ransac_result)
                T_ransac = ransac_result.transformation.astype(np.float32)
                np.savez(join(icp_save_path, filename),
                         trans=T_ransac,
                         anc_pts=anc_points,
                         pos_pts=pos_points,
                         anc_scores=anc_scores,
                         pos_scores=pos_scores
                         )
            reg_timer.toc()

            T_gth = inputs['trans']
            # loss_ransac = corr_dist(T_ransac, T_gth, anc_points, pos_points, weight=None, max_dist=1)
            loss_ransac = 0
            rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
            rre = np.arccos((np.trace(T_ransac[:3, :3].transpose() @ T_gth[:3, :3]) - 1) / 2)

            if rte < 2:
                rte_meter.update(rte)

            if not np.isnan(rre) and rre < np.pi / 180 * 5:
                rre_meter.update(rre * 180 / np.pi)

            if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
                success_meter.update(1)
            else:
                success_meter.update(0)
                logging.info(f"{anc_id} Failed with RTE: {rte}, RRE: {rre * 180 / np.pi}")

            loss_meter.update(loss_ransac)

            if (i + 1) % 10 == 0:
                logging.info(
                    f"{i+1} / {dataset.num_test}: Feat time: {feat_timer.avg}," +
                    f" Reg time: {reg_timer.avg}, Loss: {loss_meter.avg}, RTE: {rte_meter.avg}," +
                    f" RRE: {rre_meter.avg}, Success: {success_meter.sum} / {success_meter.count}" +
                    f" ({success_meter.avg * 100} %)"
                )
                feat_timer.reset()
                reg_timer.reset()

        logging.info(
            f"Total loss: {loss_meter.avg}, RTE: {rte_meter.avg}, var: {rte_meter.var}," +
            f" RRE: {rre_meter.avg}, var: {rre_meter.var}, Success: {success_meter.sum} " +
            f"/ {success_meter.count} ({success_meter.avg * 100} %)"
        )

    test_kitti(model, dset, config)
    # register each pair of fragments in scenes using multiprocessing.
    # scene_list = [
    #    '7-scenes-redkitchen',
    #    'sun3d-home_at-home_at_scan1_2013_jan_1',
    #    'sun3d-home_md-home_md_scan9_2012_sep_30',
    #    'sun3d-hotel_uc-scan3',
    #    'sun3d-hotel_umd-maryland_hotel1',
    #    'sun3d-hotel_umd-maryland_hotel3',
    #    'sun3d-mit_76_studyroom-76-1studyroom2',
    #    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    # ]
    #return_dict = Manager().dict()
    ## register_one_scene(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene_list[0])
    #jobs = []
    # for scene in scene_list:
    #    p = Process(target=register_one_scene, args=(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene))
    #    jobs.append(p)
    #    p.start()
    #
    # for proc in jobs:
    #    proc.join()
#
    #recalls = [v[0] for k, v in return_dict.items()]
    #inlier_nums = [v[1] for k, v in return_dict.items()]
    #inlier_ratios = [v[2] for k, v in return_dict.items()]
#
    #logging.info("*" * 40)
    # logging.info(recalls)
    # logging.info(f"All 8 scene, average recall: {np.mean(recalls):.2f}%")
    # logging.info(f"All 8 scene, average num inliers: {np.mean(inlier_nums):.2f}")
    # logging.info(f"All 8 scene, average num inliers ratio: {np.mean(inlier_ratios)*100:.2f}%")
