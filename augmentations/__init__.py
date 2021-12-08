
import math
import ast
import logging
import cv2
import random

import numpy as np

from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS


HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]
ARM_IDENTIFIERS_ORDER = ["neck", "$side$Shoulder", "$side$Elbow", "$side$Wrist"]


def __numpy_to_dictionary(data_array: np.ndarray) -> dict:
    output = {}
    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS):
        output[identifier] = data_array[:, landmark_index].tolist()
    return output


def __dictionary_to_numpy(landmarks_dict: dict) -> np.ndarray:
    output = np.empty(shape=(len(landmarks_dict["leftEar"]), len(BODY_IDENTIFIERS), 2))
    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS):
        output[:, landmark_index, 0] = np.array(landmarks_dict[identifier])[:, 0]
        output[:, landmark_index, 1] = np.array(landmarks_dict[identifier])[:, 1]
    return output


def __random_pass(prob):
    return random.random() < prob


def __rotate(origin: tuple, point: tuple, angle: float):
    """
    Rotates a point counterclockwise by a given angle around a given origin.

    :param origin: Landmark in the (X, Y) format of the origin from which to count angle of rotation
    :param point: Landmark in the (X, Y) format to be rotated
    :param angle: Angle under which the point shall be rotated
    :return: New landmarks (coordinates)
    """

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy


def __preprocess_row_sign(sign: dict) -> (dict, dict):
    """

    """

    # sign_eval = {key: ast.literal_eval(value) for key, value in sign.items()}
    sign_eval = sign

    if "nose_X" in sign_eval:
        body_landmarks = {identifier: [(x, y) for x, y in zip(sign_eval[identifier + "_X"], sign_eval[identifier + "_Y"])]
                          for identifier in BODY_IDENTIFIERS}
        hand_landmarks = {identifier: [(x, y) for x, y in zip(sign_eval[identifier + "_X"], sign_eval[identifier + "_Y"])]
                          for identifier in HAND_IDENTIFIERS}

    else:
        body_landmarks = {identifier: sign_eval[identifier] for identifier in BODY_IDENTIFIERS}
        hand_landmarks = {identifier: sign_eval[identifier] for identifier in HAND_IDENTIFIERS}

    return body_landmarks, hand_landmarks


def __wrap_sign_into_row(body_identifiers: dict, hand_identifiers: dict) -> dict:
    """

    """

    return {**body_identifiers, **hand_identifiers}


def augment_rotate(sign: dict, angle_range: tuple) -> dict:
    """

    """

    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)
    angle = math.radians(random.uniform(*angle_range))

    body_landmarks = {key: [__rotate((0.5, 0.5), frame, angle) for frame in value] for key, value in body_landmarks.items()}
    hand_landmarks = {key: [__rotate((0.5, 0.5), frame, angle) for frame in value] for key, value in hand_landmarks.items()}

    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


def augment_shear(sign: dict, type: str, squeeze_ratio: tuple) -> dict:
    """

    """

    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)

    if type == "squeeze":
        move_left = random.uniform(*squeeze_ratio)
        move_right = random.uniform(*squeeze_ratio)

        src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)
        dest = np.array(((0 + move_left, 1), (1 - move_right, 1), (0 + move_left, 0), (1 - move_right, 0)), dtype=np.float32)
        mtx = cv2.getPerspectiveTransform(src, dest)

    elif type == "perspective":

        move_ratio = random.uniform(*squeeze_ratio)
        src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)

        if __random_pass(0.5):
            dest = np.array(((0 + move_ratio, 1 - move_ratio), (1, 1), (0 + move_ratio, 0 + move_ratio), (1, 0)), dtype=np.float32)
        else:
            dest = np.array(((0, 1), (1 - move_ratio, 1 - move_ratio), (0, 0), (1 - move_ratio, 0 + move_ratio)), dtype=np.float32)

        mtx = cv2.getPerspectiveTransform(src, dest)

    else:

        logging.error("Unsupported shear type provided.")
        return {}

    landmarks_array = __dictionary_to_numpy(body_landmarks)
    augmented_landmarks = cv2.perspectiveTransform(np.array(landmarks_array, dtype=np.float32), mtx)

    augmented_zero_landmark = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), mtx)[0][0]
    augmented_landmarks = np.stack([np.where(sub == augmented_zero_landmark, [0, 0], sub) for sub in augmented_landmarks])

    body_landmarks = __numpy_to_dictionary(augmented_landmarks)

    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


def augment_arm_joint_rotate(sign: dict, probability: float, angle_range: tuple) -> dict:
    """

    """

    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)

    # Iterate over both directions (both hands)
    for side in ["left", "right"]:
        # Iterate gradually over the landmarks on arm
        for landmark_index, landmark_origin in enumerate(ARM_IDENTIFIERS_ORDER):
            landmark_origin = landmark_origin.replace("$side$", side)

            # End the process on the current hand if the landmark is not present
            if landmark_origin not in body_landmarks:
                break

            # Perform rotation by provided probability
            if __random_pass(probability):
                angle = math.radians(random.uniform(*angle_range))

                for to_be_rotated in ARM_IDENTIFIERS_ORDER[landmark_index + 1:]:
                    to_be_rotated = to_be_rotated.replace("$side$", side)

                    # Skip if the landmark is not present
                    if to_be_rotated not in body_landmarks:
                        continue

                    body_landmarks[to_be_rotated] = [__rotate(body_landmarks[landmark_origin][frame_index], frame, angle) for frame_index, frame in enumerate(body_landmarks[to_be_rotated])]

    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


if __name__ == "__main__":
    pass
