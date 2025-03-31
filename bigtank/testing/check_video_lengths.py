import cv2
from pathlib import Path
import subprocess as sp

import pandas as pd


def count_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return -1  # Error opening video file

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video
        frame_count += 1

    video_capture.release()
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

data_dir = Path('../../projects/thinkedge_stress_testing')
vid_paths = list(data_dir.glob('*.mp4'))
df = []

for vp in vid_paths:
    print(f'processing {vp.stem}')
    cam_id = vp.stem.split('-')[0]
    frame_count = count_frames(vp)
    start_timestamp = get_start_timestamp(vp)
    framerate = get_framerate(vp)
    t0, t1 = [eval(x) for x in vp.stem.split('T')[-1].split('-')]
    df.append({'cam_id': cam_id,
               'frame_count': frame_count,
               'start_timestamp': start_timestamp,
               'framerate': framerate,
               't0': t0,
               't1': t1})

df = pd.DataFrame(df)
df = df.sort_values(by='t0')
df.to_csv(data_dir/'summary.csv')

