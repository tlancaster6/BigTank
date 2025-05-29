import subprocess as sp
from pathlib import Path

def rotate_180(video_path: Path):
    out_path = video_path.with_name(video_path.stem + '_rotated.mp4')
    command = ['ffmpeg', '-i', str(video_path), '-vf', 'rotate=PI:bilinear=0,format=yuv420p',
               '-metadata:s:v', 'rotate=0', '-codec:v', 'libx264', '-crf', '17', '-codec:a', 'copy', str(out_path)]
    out = sp.run(command, capture_output=True, text=True)
    if out.stderr or (out.returncode != 0):
        print(f'error encountered while flipping {video_path.name}')
        print(out.stderr)

data_dir = Path('../../projects/YH_Pilot_Data')
video_paths = list(data_dir.glob('*.mp4'))
for vp in video_paths:
    print(f'processing {vp.name}')
    rotate_180(vp)



