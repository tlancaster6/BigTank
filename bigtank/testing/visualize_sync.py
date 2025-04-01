import cv2
from pathlib import Path
import subprocess as sp
import matplotlib.pyplot as plt
import pandas as pd


def uniform_sample(video_path, step=60):
    frame_dict = {}
    video_capture = cv2.VideoCapture(video_path)
    frame = None
    current_index = 0
    while True:
        previous_frame = frame
        ret, frame = video_capture.read()
        if not ret:
            frame_dict.update({current_index-1: previous_frame})
            break  # End of video
        elif current_index % step == 0:
            frame_dict.update({current_index: frame})
        current_index += 1
    video_capture.release()
    return frame_dict

data_dir = Path('../../projects/sync_testing_2')
vid_paths = sorted(list(data_dir.glob('*.mp4')))
all_frames = {}
n_cols = 0
for vp in vid_paths:
    print(f'processing {vp.stem}')
    new_frames = uniform_sample(vp, step=3600)
    all_frames.update({vp.stem: new_frames})
    n_cols = max(n_cols, len(new_frames))

fig, ax = plt.subplots(len(vid_paths), n_cols, figsize=(10*n_cols, 6*len(vid_paths)))
for i, (vid_id, frame_dict) in enumerate(all_frames.items()):
    print(f'plotting {vid_id}')
    for j, (frame_count, frame) in enumerate(frame_dict.items()):
        ax[i, j].imshow(frame)
        ax[i, j].set(title=frame_count)
fig.tight_layout()
fig.savefig(str(data_dir / 'sync_test.pdf'))
plt.close(fig)


