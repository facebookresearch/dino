#!/usr/bin/env python3

import rospy
import sys
import numpy as np
sys.path.append('/home/philipp/catkin_ws/src')
import cv2
from pathlib import Path
import os
from typing import Tuple, List
import random
import tf.transformations as tr
from tf.transformations import quaternion_from_euler, euler_from_quaternion, euler_from_matrix
import pickle
from PIL import Image
from torchvision import transforms
import copy

class PathVelocityClass():
    def __init__(self, goal_list, query_list, vel_vec):
        self.target_list = goal_list
        self.query_list = query_list
        self.vel_vec = vel_vec

class ImagePathLoader():
    def __init__(self, from_harddive, without_scaling_factors = True):
        self.from_harddive = from_harddive
        self.length_stored_vec = 8
        self.length_train_vec = 8
        if without_scaling_factors:
            self.length_train_vec = 6
        self.rgb_file_names = 'rgb_paths.pkl'
        self.target_file_names = 'target_paths.pkl'
        self.h_cone = 2
        self.theta_cone = 45
        self.rmax = 1
        self.data_path = None
        self.transform = None

    def calc_and_store_velocities(self):
        if self.from_harddive:
            parameter_scenes_paths = self.get_dataset_paths()
            for control_paths in parameter_scenes_paths:
                if 'reversed' in control_paths:
                    reversed = True
                else:
                    reversed= False
                all_trajectories = len(os.listdir(control_paths))
                for i in range(all_trajectories):
                    scene_path = os.path.join(control_paths, f'scene_{i+1}')
                    self.get_velocity(scene_path, i, reversed=reversed, save_array=True)

    def get_splitted_target_pose(self, suffix, idx, reversed=False, poses_bool = False):
        splitted_lines = None
        poses_path = self.groundtruth_path
        if poses_bool:
            poses_path = os.path.join(poses_path, f'h_{self.h_cone}_theta_{self.theta_cone}_rmax_{self.rmax}')
        file_path = os.path.join(poses_path, 'poses_' + suffix)
        if reversed:
            file_path += '_reversed'
        file_path += '.txt'
        with open(file_path) as file:
                lines = file.read()
                splitted_lines = lines.split('\n')
        return np.fromstring(splitted_lines[idx], dtype=np.float32, sep=',')[1:]

    def get_images_array(self, path, idx, reversed=False, save_array=False):
        splitted_lines = None
        poses_path = path
        file_path = os.path.join(poses_path, 'poses_groundtruth_time.txt')
        with open(file_path) as file:
                lines = file.read()
                splitted_lines = lines.split('\n')
        target_pose = self.get_splitted_target_pose(suffix='target', idx=idx, reversed=reversed, poses_bool=True)
        velocity_array = np.empty((0,self.length_train_vec))
        for line in splitted_lines:
            if line.strip():
                current_pose = np.fromstring(line, dtype=np.float32, sep=',')[1:]
                velocity_and_scale = self.calculate_pose_error(current_pose, target_pose)
                velocity_array = np.vstack([velocity_array, velocity_and_scale])
        if save_array:
            velocity_path = os.path.join(poses_path,'velocities_and_scale.npy')
            np.save(velocity_path, velocity_array)
            # velocity_txt_path = os.path.join(poses_path,'velocities_and_scale.txt')
            # np.savetxt(velocity_txt_path, velocity_array)
        return velocity_array

    def write_list_to_pickle(self, path, path_list):
        with open(path, 'wb') as file:
            pickle.dump(path_list, file)

    def read_list_from_pickle(self, path):
        path_list = None
        with open(path, 'rb') as file:
            path_list = pickle.load(file)
        return path_list

    def transform_for_check(self, im):
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ])
        im = self.transform(im)
        return im

    def is_image_corrupted(self, file_path):
        try:
            im = Image.open(file_path)
            # im.verify()
            tensor_im = self.transform_for_check(copy.deepcopy(im))
            return False
        except Exception as e:
            rospy.logerr("\033[91m" + f"Error verifying image {file_path}: {e}" + "\033[0m")
            return True

    def store_or_load_paths(self, test_stop_id=100, test_stop=False, one_direction=False):
        all_parameter_scenes_paths = self.get_dataset_paths()
        for parameter_scenes_paths in all_parameter_scenes_paths:
            num_trajectories = len(os.listdir(parameter_scenes_paths))
            if test_stop:
                num_trajectories = test_stop_id
            for i in range(num_trajectories):
                corrupt_list = np.zeros((0, 1), dtype=int)
                scene_path = os.path.join(parameter_scenes_paths, f'scene_{i+1}')
                stored_rgb_paths = os.path.join(scene_path, self.rgb_file_names)
                stored_target_paths = os.path.join(scene_path, self.target_file_names)
                
                query_list = []
                target_list = []
                
                rgb_path = os.path.join(scene_path, 'rgb')
                for path in os.listdir(rgb_path):
                    rgb_file_path = os.path.join(rgb_path, path)
                    if self.is_image_corrupted(rgb_file_path):
                        corrupt_list = np.vstack([corrupt_list, int(path.split('.')[0])])
                    else:
                        query_list.append(rgb_file_path)
                        target_list.append(os.path.join(scene_path, 'target.png'))
                query_list.append(os.path.join(scene_path, 'target.png'))
                target_list.append(os.path.join(scene_path, 'target.png'))
                self.write_list_to_pickle(stored_rgb_paths, query_list)
                self.write_list_to_pickle(stored_target_paths, target_list)
                del query_list
                del target_list
                self.get_velocity(scene_path, i, reversed=reversed, save_array=True, corrupt_list=corrupt_list)
            if one_direction:
                break

    def get_dataset_paths(self):
        if self.from_harddive:
            if self.data_path is not None:
                return parameter_scenes_paths
            media_path = '/media/philipp'
            harddrive_path = [os.path.join(media_path, entry) for entry in os.listdir(media_path)][0]
            servoing_path = os.path.join(harddrive_path, 'visual_servoing')
            local_servoing_path = os.path.join(os.getcwd(), 'src/visual_servoing')
            self.groundtruth_path = os.path.join(local_servoing_path, 'groundtruth')
            scenes_path = os.path.join(servoing_path, 'scenes')
            geometry_path = os.path.join(scenes_path, f'h_{self.h_cone}_theta_{self.theta_cone}_rmax_{self.rmax}')
            parameter_scenes_paths = [os.path.join(geometry_path, entry) for entry in os.listdir(geometry_path)]
            self.data_path = parameter_scenes_paths
        return parameter_scenes_paths

    def load_paths_and_velocities_for_training(self, perc_train_traj, without_scaling_factors = True, test_stop_id=100, test_stop=False, one_direction=False):
        train_query_list = []
        train_target_list = []
        train_vel_list = []
        train_vel_array = np.empty((0, self.length_train_vec))
        test_query_list = []
        test_target_list  = []
        test_vel_list = []
        test_vel_array = np.empty((0, self.length_train_vec))
        all_parameter_scenes_paths = self.get_dataset_paths()
        for parameter_scenes_paths in all_parameter_scenes_paths:
            num_all_traj = len(os.listdir(parameter_scenes_paths))
            if test_stop:
                num_all_traj = test_stop_id
            num_train_traj = int(num_all_traj * perc_train_traj / 100)
            for i in range(num_all_traj):
                scene_path = os.path.join(parameter_scenes_paths, f'scene_{i+1}')
                if i <= num_train_traj:
                    train_target_list.extend(self.read_list_from_pickle(os.path.join(scene_path, self.target_file_names)))
                    train_query_list.extend(self.read_list_from_pickle(os.path.join(scene_path, self.rgb_file_names)))
                    train_vel_array = np.vstack([train_vel_array, self.get_velocities_as_array(scene_path, 
                                                                                            i, 
                                                                                            from_np_file=True, 
                                                                                            without_scaling_factors=without_scaling_factors)])
                    # train_vel_list.extend(self.get_velocities_as_list(scene_path, i, from_np_file=True))
                else:
                    test_target_list.extend(self.read_list_from_pickle(os.path.join(scene_path, self.target_file_names)))
                    test_query_list.extend(self.read_list_from_pickle(os.path.join(scene_path, self.rgb_file_names)))
                    test_vel_array = np.vstack([test_vel_array, self.get_velocities_as_array(scene_path, 
                                                                                            i, 
                                                                                            from_np_file=True, 
                                                                                            without_scaling_factors=without_scaling_factors)])
                    # test_vel_list.extend(self.get_velocities_as_list(scene_path, i, from_np_file=True))
                if (num_all_traj == num_train_traj) & (i % 10 == 0):
                    test_target_list.extend(self.read_list_from_pickle(os.path.join(scene_path, self.target_file_names)))
                    test_query_list.extend(self.read_list_from_pickle(os.path.join(scene_path, self.rgb_file_names)))
                    test_vel_array = np.vstack([test_vel_array, self.get_velocities_as_array(scene_path,
                                                                                            i,
                                                                                            from_np_file=True,
                                                                                            without_scaling_factors=without_scaling_factors)])
                    # test_vel_list.extend(self.get_velocities_as_list(scene_path, i, from_np_file=True))
            if one_direction:
                break
        train_storer = PathVelocityClass(train_target_list, train_query_list, train_vel_array)
        test_storer = PathVelocityClass(test_target_list, test_query_list, test_vel_array)

        return train_storer, test_storer


    def get_velocity(self, path, idx, reversed=False, save_array=False, corrupt_list=None):
        splitted_lines = None
        poses_path = path
        file_path = os.path.join(poses_path, 'poses_groundtruth_time.txt')
        with open(file_path) as file:
                lines = file.read()
                splitted_lines = lines.split('\n')
        target_pose = self.get_splitted_target_pose(suffix='target', idx=idx, poses_bool=True)
        velocity_array = np.empty((0,self.length_stored_vec))
        for i, line in enumerate(splitted_lines, 1):
            if line.strip():
                if not np.isin(i, corrupt_list):
                    current_pose = np.fromstring(line, dtype=np.float32, sep=',')[1:]
                    velocity_and_scale = self.calculate_pose_error(current_pose, target_pose)
                    velocity_array = np.vstack([velocity_array, velocity_and_scale])
        velocity_and_scale = self.calculate_pose_error(target_pose, target_pose)
        velocity_array = np.vstack([velocity_array, velocity_and_scale])
        if save_array:
            velocity_path = os.path.join(poses_path,'velocities_and_scale.npy')
            np.save(velocity_path, velocity_array)
            # velocity_txt_path = os.path.join(poses_path,'velocities_and_scale.txt')
            # np.savetxt(velocity_txt_path, velocity_array)
        return velocity_array

    def get_velocities_as_array(self, path, idx, from_np_file=False, without_scaling_factors = True):
        if from_np_file:
            velocity_path = os.path.join(path,'velocities_and_scale.npy')
            loaded_velocity_array = np.load(velocity_path)
            if without_scaling_factors:
                velocity_array = np.hstack([loaded_velocity_array[:, :3], loaded_velocity_array[:, 4:7]])
            else:
                velocity_array = loaded_velocity_array
        return velocity_array

    def get_velocities_as_list(self, path, idx, from_np_file=False):
        if from_np_file:
            velocity_path = os.path.join(path,'velocities_and_scale.npy')
            velocity_array = np.load(velocity_path)
        else:
            velocity_array = self.get_velocity(path, idx)
        velocity_list = velocity_array.tolist()
        print(f"{idx=}_{velocity_list=}")
        return velocity_list

    def calculate_pose_error(self, current_pose, target_pose):
        # Convert current and target poses to numpy arrays for easier manipulation

        curr_pos = current_pose[:3]
        curr_quat = current_pose[3:]
        target_pos = target_pose[:3]
        target_quat = target_pose[3:]

        # Compute the position error as the difference between current and target positions
        pos_error = target_pos - curr_pos
        pos_error_thresh = 0.15
        trans_scale = np.linalg.norm(pos_error)/pos_error_thresh
        if trans_scale > 1:
            trans_scale = 1

        # Compute the orientation error as the difference between current and target quaternions
        # Convert the quaternions to rotation matrices for easier manipulation
        curr_rot = tr.quaternion_matrix(curr_quat)
        target_rot = tr.quaternion_matrix(target_quat)
        rot_error_matrix = np.dot(target_rot, curr_rot.transpose())
        rot_error_euler = tr.euler_from_matrix(rot_error_matrix)

        rot_error_thresh = 7
        ori_scale = 180/np.pi*np.linalg.norm(rot_error_euler)/rot_error_thresh
        if ori_scale > 1:
            ori_scale = 1
        # pose_error_array = np.hstack((pos_error, rot_error_quat, scale))
        pose_error_array = np.hstack([pos_error, trans_scale, rot_error_euler, ori_scale])

        return pose_error_array

if __name__ == '__main__':
    image_shuffeler= ImagePathLoader(True)
    # image_shuffeler.shuffle_paths()
    # image_shuffeler.calc_and_store_velocities()
    image_shuffeler.store_or_load_paths(test_stop=True)