# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Template for AnchorGenerator
"""

import numpy as np
import torch

from opencood.utils import box_utils


class BasePostprocessor(object):
    """
    Template for Anchor generator.

    Parameters
    ----------
    anchor_params : dict
        The dictionary containing all anchor-related parameters.
    train : bool
        Indicate train or test mode.

    Attributes
    ----------
    bbx_dict : dictionary
        Contain all objects information across the cav, key: id, value: bbx
        coordinates (1, 7)
    """

    def __init__(self, anchor_params, class_names, train=True):
        self.params = anchor_params
        self.class_names = class_names
        self.bbx_dict = {}
        self.train = train

    def generate_anchor_box(self):
        # needs to be overloaded
        return None

    def generate_label(self, *argv):
        return None

    def generate_gt_bbx(self, data_dict):
        """
        The base postprocessor will generate 3d groundtruth bounding box.
        NOTE: 将多智能体的 3D 检测框统一到 ego 坐标系下，生成最终的真实框 gt 和标签

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        Returns
        -------
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor, shape (N, 8, 3).
        """
        gt_box3d_list = []
        label_list = []
        # used to avoid repetitive bounding box
        object_id_list = []     

        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space
            # NOTE: 将真实框（gt）投影至 ego 坐标系
            transformation_matrix = cav_content['transformation_matrix']

            object_bbx_center = cav_content['object_bbx_center']
            object_bbx_mask = cav_content['object_bbx_mask']

            # NOTE: 区分于 V2V4Real，此处的 object_ids 是每辆车框对应的全局唯一 id
            object_ids = cav_content['object_ids']

            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            labels = object_bbx_center[:, -1]
            # convert center to corner
            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)
            label_list.append(labels)

            # append the corresponding ids
            object_id_list += object_ids

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        label_list = torch.cat(label_list)

        # some of the bbx may be repetitive, use the id list to filter
        # NOTE：基于 id 进行去重
        unique_object_ids = [x for x in set(object_id_list)]    # 唯一性的车辆 id 列表

        gt_box3d_selected_indices = \
            [object_id_list.index(x) for x in set(object_id_list)]  
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]
        gt_label_tensor = label_list[gt_box3d_selected_indices]

        # filter the gt_box to make sure all bbx are in the range
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)

        gt_box3d_tensor = gt_box3d_tensor[mask, :, :]
        gt_label_tensor = gt_label_tensor[mask]

        # 返回去重后的 id 列表
        unique_object_ids = [unique_object_ids[i] for i in range(len(mask)) if mask[i]]

        return gt_box3d_tensor, gt_label_tensor, unique_object_ids
    
    def generate_cp_gt_bbx(self, data_dict):
        """
        Generate CP ground truth bounding boxes using only CP's local coordinate system.
        Returns CP's local object IDs (the actual keys from vehicles dict).

        使用协同局部坐标系生成协同视角下的真实框，不涉及其他智能体及坐标系变换
        """
        gt_box3d_list = []
        label_list = []
        object_id_list = []  

        # 只关注协同视角
        cav_content = data_dict['1']
    
        # Get vehicles info to access local object IDs
        # vehicles_info = cav_content.get('vehicles_info', {})
        # if not vehicles_info:
        #     device = torch.device('cpu')
        #     if 'object_bbx_center' in cav_content:
        #         device = cav_content['object_bbx_center'].device
        #     return torch.empty(0, 8, 3, device=device), torch.empty(0, 8, 3, device=device), []
        
        # Get the raw data
        object_bbx_center = cav_content['object_bbx_center']  # (1, Max_N, 7)
        object_bbx_mask = cav_content['object_bbx_mask']     # (1, Max_N,)
        
        # Remove batch dimension
        if len(object_bbx_center.shape) == 3:
            object_bbx_center = object_bbx_center.squeeze(0)
        if len(object_bbx_mask.shape) == 2:
            object_bbx_mask = object_bbx_mask.squeeze(0)
        
        # Convert mask to numpy for boolean indexing
        if isinstance(object_bbx_mask, torch.Tensor):
            valid_mask = (object_bbx_mask == 1).cpu().numpy()
            object_bbx_center_valid = object_bbx_center[valid_mask]
        else:
            valid_mask = (object_bbx_mask == 1)
            object_bbx_center_valid = object_bbx_center[valid_mask]
        
        if object_bbx_center_valid.shape[0] == 0:
            device = object_bbx_center.device if hasattr(object_bbx_center, 'device') else torch.device('cpu')
            return torch.empty(0, 8, 3, device=device), torch.empty(0, 8, 3, device=device), []

        # The key insight: use vehicles_info keys as the local object IDs
        # The order should match the valid objects
        # local_object_ids = list(vehicles_info.keys())

        local_object_ids = cav_content['object_ids']
        
        # TODO: 这里是不是有问题，直接截取？
        # # Filter to only include valid objects (same count as valid boxes)
        # valid_local_object_ids = local_object_ids[:object_bbx_center_valid.shape[0]]
        valid_local_object_ids = local_object_ids
        
        # Convert to corner format
        if isinstance(object_bbx_center_valid, torch.Tensor):
            object_bbx_center_valid_np = object_bbx_center_valid.cpu().numpy()
        else:
            object_bbx_center_valid_np = object_bbx_center_valid

        labels = object_bbx_center_valid[:, -1]
        label_list.append(labels)
        label_list = torch.cat(label_list)
            
        gt_box3d_corner = box_utils.boxes_to_corners_3d(
            object_bbx_center_valid_np, 
            self.params['order']
        )
        
        gt_box3d_tensor = torch.from_numpy(gt_box3d_corner).to(object_bbx_center.device)

        print(len(gt_box3d_tensor), len(valid_local_object_ids))
        
        # Range filtering
        mask_within_range = box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
        gt_box3d_tensor = gt_box3d_tensor[mask_within_range]
        filtered_object_ids = [int(valid_local_object_ids[i]) for i in range(len(valid_local_object_ids)) if mask_within_range[i]]
        
        # gt_label_tensor = label_list[filtered_object_ids]
        gt_label_tensor = label_list[mask_within_range]

        return gt_box3d_tensor, gt_label_tensor, filtered_object_ids

    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 8), where 8 represents
        x, y, z, l, w, h, yaw, class or x, y, z, h, w, l, yaw, class.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 8).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        from opencood.data_utils.datasets import GT_RANGE

        tmp_object_dict = {}
        for cav_content in cav_contents:
            tmp_object_dict.update(cav_content['params']['vehicles'])

            # inserted vehicles in vehicle list
            # print("tmp ", cav_content['params']['inserted_ids'], cav_content['params']['vehicles'].keys())

        output_dict = {}
        # during training using cav_lidar_range to set learning targets
        # during inference using gt_range to set ground truth
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else GT_RANGE

        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'])
        
        
        object_np = np.zeros((self.params['max_num'], 8))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)

        return object_np, mask, object_ids
