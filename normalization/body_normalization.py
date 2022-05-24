
import logging
import pandas as pd

BODY_IDENTIFIERS = [
    "nose",
    "neck",
    "rightEye",
    "leftEye",
    "rightEar",
    "leftEar",
    "rightShoulder",
    "leftShoulder",
    "rightElbow",
    "leftElbow",
    "rightWrist",
    "leftWrist"
]


def normalize_body_full(df: pd.DataFrame) -> (pd.DataFrame, list):
    """
    Normalizes the body position data using the Bohacek-normalization algorithm.

    :param df: pd.DataFrame to be normalized
    :return: pd.DataFrame with normalized values for body pose
    """

    # TODO: Fix division by zero

    normalized_df = pd.DataFrame(columns=df.columns)
    invalid_row_indexes = []
    body_landmarks = {"X": [], "Y": []}

    # Construct the relevant identifiers
    for identifier in BODY_IDENTIFIERS:
        body_landmarks["X"].append(identifier + "_X")
        body_landmarks["Y"].append(identifier + "_Y")

    # Iterate over all of the records in the dataset
    for index, row in df.iterrows():

        sequence_size = len(row["leftEar_Y"])
        valid_sequence = True
        original_row = row

        last_starting_point, last_ending_point = None, None

        # Treat each element of the sequence (analyzed frame) individually
        for sequence_index in range(sequence_size):

            # Prevent from even starting the analysis if some necessary elements are not present
            if (row["leftShoulder_X"][sequence_index] == 0 or row["rightShoulder_X"][sequence_index] == 0) and (row["neck_X"][sequence_index] == 0 or row["nose_X"][sequence_index] == 0):
                if not last_starting_point:
                    valid_sequence = False
                    continue

                else:
                    starting_point, ending_point = last_starting_point, last_ending_point

            else:

                # NOTE:
                #
                # While in the paper, it is written that the head metric is calculated by halving the shoulder distance,
                # this is meant for the distance between the very ends of one's shoulder, as literature studying body
                # metrics and ratios generally states. The Vision Pose Estimation API, however, seems to be predicting
                # rather the center of one's shoulder. Based on our experiments and manual reviews of the data, employing
                # this as just the plain shoulder distance seems to be more corresponding to the desired metric.
                #
                # Please, review this if using other third-party pose estimation libraries.

                if row["leftShoulder_X"][sequence_index] != 0 and row["rightShoulder_X"][sequence_index] != 0:
                    left_shoulder = (row["leftShoulder_X"][sequence_index], row["leftShoulder_Y"][sequence_index])
                    right_shoulder = (row["rightShoulder_X"][sequence_index], row["rightShoulder_Y"][sequence_index])
                    shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                                (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)
                    head_metric = shoulder_distance
                else:
                    neck = (row["neck_X"][sequence_index], row["neck_Y"][sequence_index])
                    nose = (row["nose_X"][sequence_index], row["nose_Y"][sequence_index])
                    neck_nose_distance = ((((neck[0] - nose[0]) ** 2) + ((neck[1] - nose[1]) ** 2)) ** 0.5)
                    head_metric = neck_nose_distance

                # Set the starting and ending point of the normalization bounding box
                starting_point = [row["neck_X"][sequence_index] - 3 * head_metric, row["leftEye_Y"][sequence_index] + (head_metric / 2)]
                ending_point = [row["neck_X"][sequence_index] + 3 * head_metric, starting_point[1] - 6 * head_metric]

                last_starting_point, last_ending_point = starting_point, ending_point

            # Ensure that all of the bounding-box-defining coordinates are not out of the picture
            if starting_point[0] < 0: starting_point[0] = 0
            if starting_point[1] < 0: starting_point[1] = 0
            if ending_point[0] < 0: ending_point[0] = 0
            if ending_point[1] < 0: ending_point[1] = 0

            # Normalize individual landmarks and save the results
            for identifier in BODY_IDENTIFIERS:
                key = identifier + "_"

                # Prevent from trying to normalize incorrectly captured points
                if row[key + "X"][sequence_index] == 0:
                    continue

                normalized_x = (row[key + "X"][sequence_index] - starting_point[0]) / (ending_point[0] -
                                                                                       starting_point[0])
                normalized_y = (row[key + "Y"][sequence_index] - ending_point[1]) / (starting_point[1] -
                                                                                       ending_point[1])

                row[key + "X"][sequence_index] = normalized_x
                row[key + "Y"][sequence_index] = normalized_y

        if valid_sequence:
            normalized_df = normalized_df.append(row, ignore_index=True)
        else:
            logging.warning(" BODY LANDMARKS: One video instance could not be normalized.")
            normalized_df = normalized_df.append(original_row, ignore_index=True)
            invalid_row_indexes.append(index)

    print("The normalization of body is finished.")
    print("\t-> Original size:", df.shape[0])
    print("\t-> Normalized size:", normalized_df.shape[0])
    print("\t-> Problematic videos:", len(invalid_row_indexes))

    return normalized_df, invalid_row_indexes


def normalize_single_dict(row: dict):
    """
    Normalizes the skeletal data for a given sequence of frames with signer's body pose data. The normalization follows
    the definition from our paper.

    :param row: Dictionary containing key-value pairs with joint identifiers and corresponding lists (sequences) of
                that particular joints coordinates
    :return: Dictionary with normalized skeletal data (following the same schema as input data)
    """

    sequence_size = len(row["leftEar"])
    valid_sequence = True
    original_row = row

    last_starting_point, last_ending_point = None, None

    # Treat each element of the sequence (analyzed frame) individually
    for sequence_index in range(sequence_size):

        # Prevent from even starting the analysis if some necessary elements are not present
        if (row["leftShoulder"][sequence_index][0] == 0 or row["rightShoulder"][sequence_index][0] == 0) and (
                row["neck"][sequence_index][0] == 0 or row["nose"][sequence_index][0] == 0):
            if not last_starting_point:
                valid_sequence = False
                continue

            else:
                starting_point, ending_point = last_starting_point, last_ending_point

        else:

            # NOTE:
            #
            # While in the paper, it is written that the head metric is calculated by halving the shoulder distance,
            # this is meant for the distance between the very ends of one's shoulder, as literature studying body
            # metrics and ratios generally states. The Vision Pose Estimation API, however, seems to be predicting
            # rather the center of one's shoulder. Based on our experiments and manual reviews of the data, employing
            # this as just the plain shoulder distance seems to be more corresponding to the desired metric.
            #
            # Please, review this if using other third-party pose estimation libraries.

            if row["leftShoulder"][sequence_index][0] != 0 and row["rightShoulder"][sequence_index][0] != 0:
                left_shoulder = (row["leftShoulder"][sequence_index][0], row["leftShoulder"][sequence_index][1])
                right_shoulder = (row["rightShoulder"][sequence_index][0], row["rightShoulder"][sequence_index][1])
                shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                        (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)
                head_metric = shoulder_distance
            else:
                neck = (row["neck"][sequence_index][0], row["neck"][sequence_index][1])
                nose = (row["nose"][sequence_index][0], row["nose"][sequence_index][1])
                neck_nose_distance = ((((neck[0] - nose[0]) ** 2) + ((neck[1] - nose[1]) ** 2)) ** 0.5)
                head_metric = neck_nose_distance

            # Set the starting and ending point of the normalization bounding box
            #starting_point = [row["neck"][sequence_index][0] - 3 * head_metric,
            #                 row["leftEye"][sequence_index][1] + (head_metric / 2)]
            starting_point = [row["neck"][sequence_index][0] - 1 * head_metric,
                            row["leftEye"][sequence_index][1] - head_metric/2]
            ending_point = [row["neck"][sequence_index][0] + 1 * head_metric,
                            starting_point[1] + 3 * head_metric]

            last_starting_point, last_ending_point = starting_point, ending_point

        # Ensure that all of the bounding-box-defining coordinates are not out of the picture
        if starting_point[0] < 0: starting_point[0] = 0
        if starting_point[1] > 1: starting_point[1] = 1
        if ending_point[0] < 0: ending_point[0] = 0
        if ending_point[1] > 1: ending_point[1] = 1

        # Normalize individual landmarks and save the results
        for identifier in BODY_IDENTIFIERS:
            key = identifier

            # Prevent from trying to normalize incorrectly captured points
            if row[key][sequence_index][0] == 0:
                continue

            if (ending_point[0] - starting_point[0]) == 0 or (starting_point[1] - ending_point[1]) == 0:
                logging.info("Problematic normalization")
                valid_sequence = False
                break

            normalized_x = (row[key][sequence_index][0] - starting_point[0]) / (ending_point[0] - starting_point[0])
            normalized_y = (row[key][sequence_index][1] - starting_point[1]) / (ending_point[1] - starting_point[1])

            row[key][sequence_index] = list(row[key][sequence_index])

            row[key][sequence_index][0] = normalized_x
            row[key][sequence_index][1] = normalized_y

    if valid_sequence:
        return row

    else:
        return original_row


if __name__ == "__main__":
    pass
