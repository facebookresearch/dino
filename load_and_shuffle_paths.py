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

class PathVelocityClass():
    def __init__(self, goal_list, query_list, vel_vec):
        self.target_list = goal_list
        self.query_list = query_list
        self.vel_vec = vel_vec

class ImagePathLoader():
    def __init__(self, from_harddive, without_scaling_factors = True):
        self.from_harddive = from_harddive
        self.length_train_vec = 8
        if without_scaling_factors:
            self.length_train_vec = 6
        self.rgb_file_names = 'rgb_paths.pkl'
        self.target_file_names = 'target_paths.pkl'
        self.h_cone = 2
        self.theta_cone = 45

    def calc_and_store_velocities(self):
        if self.from_harddive:
            parameter_scenes_paths = self.get_dataset_path()
            all_trajectories = 600
            for i in range(all_trajectories):
                scene_path = os.path.join(parameter_scenes_paths, f'scene_{i+1}')
                self.get_velocity(scene_path, i, save_array=True)

    def get_splitted_target_pose(self, suffix, idx, poses_bool = False):
        splitted_lines = None
        poses_path = self.groundtruth_path
        if poses_bool:
            poses_path = os.path.join(poses_path, f'h_{self.h_cone}_theta_{self.theta_cone}')
        file_path = os.path.join(poses_path, 'poses_'+ suffix + '.txt')
        with open(file_path) as file:
                lines = file.read()
                splitted_lines = lines.split('\n')
        return np.fromstring(splitted_lines[idx], dtype=np.float32, sep=',')[1:]

    def get_images_array(self, path, idx, save_array=False):
        splitted_lines = None
        poses_path = path
        file_path = os.path.join(poses_path, 'poses_groundtruth_time.txt')
        with open(file_path) as file:
                lines = file.read()
                splitted_lines = lines.split('\n')
        target_pose = self.get_splitted_target_pose(suffix='target', idx=idx, poses_bool=True)
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

    def store_or_load_paths(self, num_trajectories=600):
        
        parameter_scenes_paths = self.get_dataset_path()
        for i in range(num_trajectories):
            scene_path = os.path.join(parameter_scenes_paths, f'scene_{i+1}')
            stored_rgb_paths = os.path.join(scene_path, self.rgb_file_names)
            stored_target_paths = os.path.join(scene_path, self.target_file_names)
            
            query_list = []
            target_list = []
            
            rgb_path = os.path.join(scene_path, 'rgb')
            for path in os.listdir(rgb_path):
                query_list.append(os.path.join(rgb_path, path))
                target_list.append(os.path.join(scene_path, 'target.png'))
            query_list.append(os.path.join(scene_path, 'target.png'))
            target_list.append(os.path.join(scene_path, 'target.png'))
            self.write_list_to_pickle(stored_rgb_paths, query_list)
            self.write_list_to_pickle(stored_target_paths, target_list)
            del query_list
            del target_list

    def get_dataset_path(self):
        if self.from_harddive:
            harddrive_path = '/media/philipp/PhilippUSB/visual_servoing'
            scenes_path = os.path.join(harddrive_path, 'scenes')
            self.groundtruth_path = os.path.join(harddrive_path, 'groundtruth')
            parameter_scenes_paths = os.path.join(scenes_path, f'h_{self.h_cone}_theta_{self.theta_cone}/kp_0.1_kd_0.01_vm_1_om_0.2')
        return parameter_scenes_paths

    def load_paths_and_velocities_for_training(self, num_all_traj, num_train_traj, without_scaling_factors = True):
        train_query_list = []
        train_target_list = []
        train_vel_list = []
        train_vel_array = np.empty((0, self.length_train_vec))
        test_query_list = []
        test_target_list  = []
        test_vel_list = []
        test_vel_array = np.empty((0, self.length_train_vec))
        parameter_scenes_paths = self.get_dataset_path()
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
        
        train_storer = PathVelocityClass(train_target_list, train_query_list, train_vel_array)
        test_storer = PathVelocityClass(test_target_list, test_query_list, test_vel_array)

        return train_storer, test_storer


    def get_velocity(self, path, idx, save_array=False):
        splitted_lines = None
        poses_path = path
        file_path = os.path.join(poses_path, 'poses_groundtruth_time.txt')
        with open(file_path) as file:
                lines = file.read()
                splitted_lines = lines.split('\n')
        target_pose = self.get_splitted_target_pose(suffix='target', idx=idx, poses_bool=True)
        velocity_array = np.empty((0,self.length_train_vec))
        for line in splitted_lines:
            if line.strip():
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
        # if np.linalg.norm(pos_error) > pos_error_thresh:
        #     trans_scale = 1
        # else:
        #     trans_scale = np.linalg.norm(pos_error)/pos_error_thresh
        trans_scale = np.linalg.norm(pos_error)/pos_error_thresh

        # Compute the orientation error as the difference between current and target quaternions
        # Convert the quaternions to rotation matrices for easier manipulation
        curr_rot = tr.quaternion_matrix(curr_quat)
        target_rot = tr.quaternion_matrix(target_quat)
        rot_error_matrix = np.dot(target_rot, curr_rot.transpose())
        rot_error_euler = tr.euler_from_matrix(rot_error_matrix)

        rot_error_thresh = 7
        # if 180/np.pi*np.linalg.norm(rot_error_euler) > rot_error_thresh:
        #     ori_scale = 1
        # else:
        #     ori_scale = 180/np.pi*np.linalg.norm(rot_error_euler)/rot_error_thresh
        ori_scale = 180/np.pi*np.linalg.norm(rot_error_euler)/rot_error_thresh
        # pose_error_array = np.hstack((pos_error, rot_error_quat, scale))
        pose_error_array = np.hstack([pos_error, trans_scale, rot_error_euler, ori_scale])

        return pose_error_array
            

if __name__ == '__main__':
    image_shuffeler= ImagePathLoader(True)
    # image_shuffeler.shuffle_paths()
    image_shuffeler.calc_and_store_velocities()
    # image_shuffeler.store_paths()