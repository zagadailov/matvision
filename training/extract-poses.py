"""Sample a licensed video at 4 fps using MatVision's two-athlete pose model."""

import argparse
import json
import subprocess
from pathlib import Path

import mediapipe as mp
import numpy as np


def dimensions(video):
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(video)
    ])
    stream = json.loads(output)["streams"][0]
    return stream["width"], stream["height"]


def extract(video, model_path, destination, start=0, duration=None):
    native_width, native_height = dimensions(video)
    width = min(native_width, 640)
    height = round(width * native_height / native_width / 2) * 2
    sample_rate = 4
    frame_size = width * height * 3
    joints = [11, 12, 23, 24]  # MediaPipe shoulders and hips.

    command = [
        "ffmpeg", "-v", "error", "-threads", "1", "-filter_threads", "1",
        "-ss", str(start), "-i", str(video),
        *([] if duration is None else ["-t", str(duration)]),
        "-vf", f"fps={sample_rate}:start_time=0,scale={width}:{height}",
        "-an", "-c:v", "rawvideo", "-threads:v", "1", "-pix_fmt", "rgb24",
        "-f", "rawvideo", "pipe:1"
    ]
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=2,
        min_pose_detection_confidence=0.45, min_pose_presence_confidence=0.45
    )
    rows = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
        with subprocess.Popen(command, stdout=subprocess.PIPE) as decoder:
            index = 0
            while len(pixels := decoder.stdout.read(frame_size)) == frame_size:
                rgb = np.frombuffer(pixels, np.uint8).reshape(height, width, 3)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = 1000 * index // sample_rate
                result = model.detect_for_video(image, timestamp)
                poses = [
                    [[round(pose[i].x, 5), round(pose[i].y, 5), round(pose[i].visibility, 4)]
                     for i in joints] for pose in result.pose_landmarks
                ]
                rows.append(poses)
                index += 1
            if decoder.wait() != 0:
                raise RuntimeError(f"Could not decode {video}")
    destination.write_text(json.dumps({"aspect": width / height, "sampleRate": sample_rate, "start": start, "frames": rows}))
    print(f"{video.name}: {len(rows)} frames, {sum(len(row) == 2 for row in rows)} with two pose estimates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    extract(args.video, args.model, args.output, args.start, args.duration)
