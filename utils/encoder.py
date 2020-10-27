import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform

BOX_ENLARGE = 1.3
BOX_SHIFT = 0.2
INPUT_SHAPE = (256, 256, 3)


def normalize_image(rgb_image):
    return np.ascontiguousarray(2 * ((rgb_image / 255) - 0.5).astype('float32'))


def preprocess(bgr_image):
    """
    :param bgr_image: image read by cv2
    :return: preprocessed image and padding value
    """
    rgb_image = bgr_image[:, :, ::-1]
    shape = np.int32(bgr_image.shape)
    padding = (shape.max() - shape[:2]).astype('uint32') // 2
    rgb_image = np.pad(rgb_image, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant')
    rgb_image = cv2.resize(rgb_image, INPUT_SHAPE[:2])
    rgb_image = normalize_image(rgb_image)
    rgb_image = rgb_image[tf.newaxis, ...]
    return rgb_image, padding


def max_distance(points):
    """
    :param points: 2D points, shape: (number_of_points, 2)
    :return: the max distance among all pairs of points
    """
    d = pdist(points)
    d = squareform(d)
    return np.nanmax(d)


def get_triangle(wrist, mmcp, w):
    """
    :param wrist: 2D coordinate of wrist
    :param mmcp: 2D coordinate of mmcp
    :param w: max side of bounding box or
    :return:
    """
    side = w * BOX_ENLARGE
    dir_v = mmcp - wrist
    dir_v /= np.linalg.norm(dir_v)
    dir_v_r = dir_v @ np.r_[[[0, 1], [-1, 0]]].T
    triangle = np.float32([mmcp, mmcp + dir_v * side, mmcp + dir_v_r * side])
    triangle -= (wrist - mmcp) * BOX_SHIFT
    return triangle


def encode_landmarks(landmarks, matrix):
    ones = np.ones(shape=(len(landmarks), 1))
    landmarks_ones = np.hstack([landmarks, ones])
    encoded_landmarks = matrix.dot(landmarks_ones.T).T
    return encoded_landmarks
