import cv2
from pathlib import Path
import subprocess as sp
import matplotlib.pyplot as plt

import pandas as pd


def count_frames(video_path, return_first_and_last_frames=True):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return -1  # Error opening video file

    frame_count = 0
    frame = None
    while True:
        previous_frame = frame
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video
        if frame_count == 0:
            first_frame = frame
        frame_count += 1
    last_frame = previous_frame

    video_capture.release()
    if return_first_and_last_frames:
        return frame_count, first_frame, last_frame
    else:
        return frame_count

def get_start_timestamp(video_path):
    command = ['ffprobe', '-v', '0', '-show_entries', 'format=start_time', '-of', 'compact=p=0:nk=1', str(video_path)]
    out = sp.run(command, capture_output=True, encoding='utf-8')
    return eval(out.stdout)

def get_framerate(video_path):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()
    return fps

def generate_frame_plots(first_frames_list, last_frames_list, n_cameras=4, figsize=(20, 20)):
    n_iters = len(first_frames_list) // n_cameras

    startframe_fig, startframe_axes = plt.subplots(n_cameras, n_iters, figsize=figsize)
    startframe_axes = startframe_axes.flatten()
    for i, startframe in enumerate(first_frames_list):
        startframe_axes[i].imshow(startframe)
    startframe_fig.tight_layout()
    startframe_fig.savefig(str(data_dir / 'startframe_summary.pdf'))
    plt.close(startframe_fig)

    endframe_fig, endframe_axes = plt.subplots(n_cameras, n_iters, figsize=figsize)
    endframe_axes = endframe_axes.flatten()
    for i, endframe in enumerate(last_frames_list):
        endframe_axes[i].imshow(endframe)
    endframe_fig.tight_layout()
    endframe_fig.savefig(str(data_dir / 'endframe_summary.pdf'))
    plt.close(endframe_fig)




data_dir = Path('../../projects/thinkedge_stress_testing')
vid_paths = sorted(list(data_dir.glob('*.mp4')))
df = []
first_frames = []
last_frames = []
for vp in vid_paths:
    print(f'processing {vp.stem}')
    cam_id = vp.stem.split('-')[0]
    frame_count, first_frame, last_frame = count_frames(vp)
    first_frames.append(first_frame)
    last_frames.append(last_frame)
    start_timestamp = get_start_timestamp(vp)
    framerate = get_framerate(vp)
    t0, t1 = [eval(x.lstrip('0')) for x in vp.stem.split('T')[-1].split('-')]
    df.append({'cam_id': cam_id,
               'frame_count': frame_count,
               'start_timestamp': start_timestamp,
               'framerate': framerate,
               't0': t0,
               't1': t1})

generate_frame_plots(first_frames, last_frames, n_cameras=4, figsize=(20, 20))
df = pd.DataFrame(df)
df = df.sort_values(by='t0')
df.to_csv(data_dir/'summary.csv')


