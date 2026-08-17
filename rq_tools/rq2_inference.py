import argparse
import statistics
import sys
import os

import torch
import shutil
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from rq_tools import eval_utils
from rq_tools.rq2_data_select import CooTest_method_result, V2X_Gen_method
from rq_tools.v2x_gen_utils import save_box_tensor, load_box_tensor, get_valid_param_dict, get_total_occ_and_dis
from config.common_config import VEHICLE_CLASS_ID

def rq2_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt


def main():
    opt = rq2_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']

    # if opt.fusion_method == 'nofusion':
    #     hypes = yaml_utils.load_yaml(opt.model_dir, None)
    # else:
    hypes = yaml_utils.load_yaml("", opt)

    hypes['validate_dir'] = opt.dataset_dir
    # hypes['dataset_mode'] = 'vc'

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {
        0.5: {'tp': [], 'fp': [], 'gt': []},
        0.7: {'tp': [], 'fp': [], 'gt': []},
        'occ_error': [],
        'dis_error': [],
        'total_occ': [],
        'total_dis': [],
        'timestamp': []
    }
    select_result_stat = {
        'v2x_gen': [],
        'cootest': []
    }

    total_fop = []
    total_flp = []

    total_occlusion = 0

    if opt.fusion_method == 'nofusion':
        # nofusion method

        det_save_path = "/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq2/rq2_det_box"
        
        # 如果目录已存在，则删除整个目录
        if os.path.exists(det_save_path):
            shutil.rmtree(det_save_path)
        # 重新创建目录
        os.makedirs(det_save_path, exist_ok=True)

        for i, batch_data in enumerate(data_loader):
            print('data idx =', i)

            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = train_utils.to_device(batch_data, device)

                det_box_tensor, det_score, _, _, _ = \
                    inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)

                # 只关注类别为 vehicle (key = 1) 的对象
                keep_index_det = det_score[:, -1] == VEHICLE_CLASS_ID
                det_box_tensor = det_box_tensor[keep_index_det]
                det_score = det_score[keep_index_det, 0]

                if det_box_tensor is not None:
                    save_box_tensor(det_box_tensor, det_score, i, det_save_path)
    else:
        # late/intermediate/early method
        for i, batch_data in enumerate(data_loader):
            # print('data idx =', i)
            # if i in [204]:
            #     continue
            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = train_utils.to_device(batch_data, device)

                det_save_path = "/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq2/rq2_det_box"
                det_box_tensor, det_score = load_box_tensor(i, det_save_path)

                if opt.fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor, gt_object_ids = \
                        inference_utils.inference_late_fusion(batch_data,
                                                             model,
                                                             opencood_dataset)
                elif opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor, gt_object_ids = \
                        inference_utils.inference_early_fusion(batch_data,
                                                              model,
                                                              opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor, gt_object_ids = \
                        inference_utils.inference_intermediate_fusion(batch_data,
                                                                     model,
                                                                     opencood_dataset)
                else:
                    raise NotImplementedError('Only early, late and intermediate'
                                              'fusion is supported.')
                
                # 只关注类别为 vehicle (key = 1) 的对象
                # print(type(pred_score))
                if pred_score == None:
                    print(i, type(pred_box_tensor))
                    continue
                keep_index_pred = pred_score[:, -1] == VEHICLE_CLASS_ID
                pred_box_tensor = pred_box_tensor[keep_index_pred]
                gt_box_tensor = gt_box_tensor[gt_label_tensor == VEHICLE_CLASS_ID]
                pred_score = pred_score[keep_index_pred, 0]

                # overall calculating
                fp, tp, gt, false_pred_ids = eval_utils.caluclate_tp_fp(pred_box_tensor,
                                                                        pred_score,
                                                                        gt_box_tensor,
                                                                        result_stat,
                                                                        0.5,
                                                                        gt_object_ids=gt_object_ids.copy())

                # if gt != len(gt_object_ids):
                #     sys.exit()

                ego_v2x_gen_dict = batch_data['ego']['v2x_gen']

                timestamp_key = ego_v2x_gen_dict[list(ego_v2x_gen_dict.keys())[0]]['timestamp']
                folder_name = ego_v2x_gen_dict[list(ego_v2x_gen_dict.keys())[0]]['folder_name']
                
                # scene, timestamp
                timestamp = [folder_name, timestamp_key]
                result_stat['timestamp'].append(timestamp)

                if opt.fusion_method == 'late':
                    if '1' in batch_data.keys():
                        cp_v2x_gen_dict = batch_data['1']['v2x_gen']
                        # cp_v2x_gen_dict = get_valid_param_dict(cp_v2x_gen_dict, batch_data['1']['object_ids'])
                        cp_v2x_gen_dict = get_valid_param_dict(cp_v2x_gen_dict, gt_object_ids)
                    # 没有 cp 数据
                    else:
                        # print(timestamp, folder_name)
                        cp_v2x_gen_dict = {}
                else:
                    if 'v2x_gen_cp' not in batch_data['ego']:
                        cp_v2x_gen_dict = {}
                    else:
                        cp_v2x_gen_dict = batch_data['ego']['v2x_gen_cp']
                    # cp_v2x_gen_dict = {}
                # ego_v2x_gen_dict = get_valid_param_dict(ego_v2x_gen_dict, batch_data['ego']['object_ids'])
                ego_v2x_gen_dict = get_valid_param_dict(ego_v2x_gen_dict, gt_object_ids)

                occ_error, total_occ = eval_utils.get_occ_error(ego_v2x_gen_dict, cp_v2x_gen_dict,
                                                                false_pred_ids)
                dis_error, total_dis = eval_utils.get_long_distance_error(ego_v2x_gen_dict, cp_v2x_gen_dict,
                                                                          false_pred_ids, 50)
                
                result_stat['occ_error'].append(occ_error)
                result_stat['dis_error'].append(dis_error)
                result_stat['total_occ'].append(total_occ)
                result_stat['total_dis'].append(total_dis)

                cootest_method_result = CooTest_method_result(det_box_tensor, det_score, pred_box_tensor)
                gen_method_result, fop, flp = V2X_Gen_method(ego_v2x_gen_dict, cp_v2x_gen_dict,
                                                             false_pred_ids, a=0.5, b=0.5)
                select_result_stat['v2x_gen'].append(gen_method_result)
                total_flp.append(flp)
                total_fop.append(fop)
                # print(gen_method_result)
                select_result_stat['cootest'].append(cootest_method_result)
                total_occlusion += total_occ
                print(i, 'cootest param =', cootest_method_result, ', v2x gen param =', gen_method_result)
                # print('idx = ', i, ' occ error =', occ_error, ', dis error =', dis_error, total_occ, total_dis)

    # result
    if opt.fusion_method == 'nofusion':
        eval_utils.eval_final_results(result_stat,
                                      opt.model_dir)
    else:
        print("total occlusion = ", total_occlusion)
        # eval_utils.method_eval_result(select_result_stat, result_stat, opt.model_dir, 0.1, True, opt.dataset_dir,
        #                               '/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq2/rq2_select', opt.model_dir.split('/')[-1])
        eval_utils.method_eval_result(select_result_stat, result_stat, opt.model_dir, 0.15, True, opt.dataset_dir,
                                      '/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq2/rq2_select', opt.model_dir.split('/')[-1])
        print("total FOP =", sum(total_fop), "total FLP =", sum(total_flp))
        print("avg FOP =", sum(total_fop) / len(total_fop))
        print("avg FLP =", sum(total_flp) / len(total_flp))
        print("mid FOP =", statistics.median(total_fop))
        print("mid FLP =", statistics.median(total_flp))


if __name__ == '__main__':
    main()
