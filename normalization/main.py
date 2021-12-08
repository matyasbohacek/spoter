
import ast
import pandas as pd

from normalization.hand_normalization import normalize_hands_full
from normalization.body_normalization import normalize_body_full


# Load the dataset
df = pd.read_csv("/Users/matyasbohacek/Documents/WLASL_test_15fps.csv", encoding="utf-8")

# Retrieve metadata
video_size_heights = df["video_size_height"].to_list()
video_size_widths = df["video_size_width"].to_list()

# Delete redundant (non-related) properties
del df["video_size_height"]
del df["video_size_width"]

# Temporarily remove other relevant metadata
labels = df["labels"].to_list()
video_fps = df["video_fps"].to_list()
del df["labels"]
del df["video_fps"]

# Convert the strings into lists
convert = lambda x: ast.literal_eval(str(x))
for column in df.columns:
    df[column] = df[column].apply(convert)

# Perform the normalizations
df = normalize_hands_full(df)
df, invalid_row_indexes = normalize_body_full(df)

# Clear lists of items from deleted rows
# labels = [t for i, t in enumerate(labels) if i not in invalid_row_indexes]
# video_fps = [t for i, t in enumerate(video_fps) if i not in invalid_row_indexes]

# Return the metadata back to the dataset
df["labels"] = labels
df["video_fps"] = video_fps

df.to_csv("/Users/matyasbohacek/Desktop/WLASL_test_15fps_normalized.csv", encoding="utf-8", index=False)
