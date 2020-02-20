import os
import json
import numpy as np
from skimage import io
from numpy.linalg import inv
import cv2
from scipy import ndimage
import random


# used to identify an image with the episode, camera and image number.
class ImageIndex:
    def __init__(self, episode, camera, index):
        self.episode = episode
        self.camera = camera
        self.index = index


def generate_matches_carla(benchmark_folder, all_weathers, image_index_1, image_index_2, use_dso_depths=False):
    folder = os.path.join(benchmark_folder, "all_weathers") if all_weathers else os.path.join(benchmark_folder,
                                                                                              "one_weather")
    folder_1 = os.path.join(folder, 'episode_00' + str(image_index_1.episode))
    folder_2 = os.path.join(folder, 'episode_00' + str(image_index_2.episode))

    # for image_1 only camera 0 is supported for now.
    if use_dso_depths and image_index_1.camera != 0:
        print("Error: Camera for image index 1 must be 0, as we only supply CoarseDepths for camera 0\n")
        return None

    coarse_depths_file = os.path.join(folder_1, "CoarseDepths", str(image_index_1.index) + ".txt")

    if all_weathers:
        if image_index_1.episode % 3 != image_index_2.episode % 3:
            print("Error: Episodes are following different trajectories.\n")
            return None
    else:
        if image_index_1.episode != image_index_2.episode:
            print("Error: When using same-weather correspondences both images need to be from the same episode.\n")
            return None

    img1, depth1, pose1, intrinsics = load_data(folder_1, image_index_1.camera, image_index_1.index)
    img2, depth2, pose2, intrinsics = load_data(folder_2, image_index_2.camera, image_index_2.index)

    if use_dso_depths:
        if not (os.path.exists(coarse_depths_file)):
            print(
                "Error: Coarse depths file does not exist. Note that Coarse depths are only available for some images.\n")
            return None
        pointcloud = load_pointcloud(coarse_depths_file)
        matches = get_matches_dso_points(pointcloud, np.dot(inv(pose2), pose1), intrinsics, img1.shape)
    else:
        # matches have the form [x, y, x', y']
        matches = get_correspondences(img1, img2, depth1, depth2, inv(pose1), inv(pose2), intrinsics)

    return matches, img1, img2


def load_data(episode_folder, camera, index):
    posefile = os.path.join(episode_folder, "transforms.json")
    with open(posefile) as f:
        posedata = json.load(f)

    intrinsic_file = os.path.join(episode_folder, 'camera_intrinsic.json')
    with open(intrinsic_file) as f:
        cam_intrinsic = json.load(f)

    img_file = os.path.join(episode_folder, "CameraRGB{}/image_{:05d}.png".format(camera, index))
    depth_file = os.path.join(episode_folder, "CameraDepth{}/image_{:05d}.png".format(camera, index))

    pose = np.array(posedata["image_{:05d}".format(index)][camera])
    img = io.imread(img_file)

    # the groundtruth-depth file format is documented in https://carla.readthedocs.io/en/stable/cameras_and_sensors/
    depth = io.imread(depth_file)
    depth = depth[:, :, 0] * 1.0 + depth[:, :, 1] * 256.0 + depth[:, :, 2] * (256.0 * 256.0)
    depth = depth * (1000.0 / (256.0 * 256.0 * 256.0 - 1.0))

    return img, depth, pose, np.array(cam_intrinsic)


# for each pyramid level we output [depths, u, v]
def load_pointcloud(filename):
    pc = np.loadtxt(filename, delimiter=" ")

    ret = list()

    max_lvl = pc[:, 0].max()
    for lvl in range(int(max_lvl) + 1):
        indices = pc[:, 0] == lvl
        pc_lvl = np.concatenate([np.expand_dims(1.0 / pc[indices, 3], 1), pc[indices, 1:3]], axis=1)
        ret.append(pc_lvl)
    return ret


def get_correspondences(img1, img2, depth1m, depth2m, world_to_cam1, world_to_cam2, cam_intrinsic):
    # parameters
    filter_depth_at = 900
    cut_boundaries = 5
    num_matches = 2000

    image_size = img1.shape

    cam1_to_cam2 = np.dot(world_to_cam2, inv(world_to_cam1))
    cam2_to_cam1 = np.dot(world_to_cam1, inv(world_to_cam2))

    assert depth1m.shape == depth2m.shape
    assert img1.shape == img2.shape

    # init correspondences
    correspondences = []

    # threshold image
    gray_image = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    dx = ndimage.sobel(gray_image, axis=0, mode='constant')
    dy = ndimage.sobel(gray_image, axis=1, mode='constant')
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    ind_y, ind_x = np.where(mag > 10)

    # create list and randomly shuffle
    ind_list = list(zip(ind_y, ind_x))
    random.shuffle(ind_list)

    # for all indices
    for y, x in ind_list:
        depth = depth1m[y, x]

        # check depth value and clip it
        if depth > filter_depth_at:
            continue

        # check that depth is large enough
        if depth > 0.1:

            ox, oy = transfer_coordinate(inv(cam_intrinsic), cam_intrinsic, cam1_to_cam2, x, y, depth)
            # round values
            rox = int(round(ox))
            roy = int(round(oy))

            # check for boundaries
            if cut_boundaries <= ox <= image_size[
                1] - cut_boundaries - 1 and cut_boundaries <= oy <= image_size[
                0] - cut_boundaries - 1 and cut_boundaries <= x <= image_size[
                1] - cut_boundaries - 1 and cut_boundaries <= y <= image_size[
                0] - cut_boundaries - 1:

                odepth = depth2m[roy, rox]

                if odepth > 0.1:
                    bx, by = transfer_coordinate(inv(cam_intrinsic), cam_intrinsic, cam2_to_cam1, ox, oy, odepth)

                    # check if the correspondence is off
                    # if abs(bx - x) + abs(by - y) >= 1:
                    if np.linalg.norm(np.array((bx, by)) - np.array((x, y))) > 2:
                        continue

                    # append found correspondences
                    correspondences.append([x, y, ox, oy])

                    if len(correspondences) >= num_matches:
                        return np.array(correspondences)

    return np.array(correspondences)


def transfer_coordinate(intrinsics1_inv, intrinsics2, cam1_to_cam2, x, y, depth):
    coords = np.array([x, y, 1]) * depth

    point3d = np.append(np.dot(intrinsics1_inv, coords), [1], axis=0)

    # 3d point in coords of second camera
    point3d2 = np.dot(cam1_to_cam2, point3d)

    coords2d = np.dot(intrinsics2, point3d2[0:3])

    # normalize the point
    ox = coords2d[0] / coords2d[2]
    oy = coords2d[1] / coords2d[2]

    return ox, oy


# we want to output first the largest image correspondences, then the smaller ones.
def get_matches_dso_points(pointcloud, cam0_to_cam1, intrinsics, shape):
    cut_boundaries = 5

    curr_points = pointcloud[0]
    intrinsics_inv = inv(intrinsics)

    num_points = curr_points.shape[0]
    correspondences = []

    for i in range(num_points):
        x = curr_points[i, 1]
        y = curr_points[i, 2]
        depth = curr_points[i, 0]

        ox, oy = transfer_coordinate(intrinsics_inv, intrinsics, cam0_to_cam1, x, y, depth)

        if cut_boundaries <= ox <= shape[
            1] - cut_boundaries - 1 and cut_boundaries <= oy <= shape[
            0] - cut_boundaries - 1 and cut_boundaries <= x <= shape[
            1] - cut_boundaries - 1 and cut_boundaries <= y <= shape[
            0] - cut_boundaries - 1:
            correspondences.append([x, y, ox, oy])

    return np.array(correspondences)
