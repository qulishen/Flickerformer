#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".mpeg", ".mpg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="从指定文件夹中批量抽取视频帧，并按视频名保存到对应子文件夹。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="输入视频文件夹，例如包含 0001.mp4、0002.mp4 等文件。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出文件夹，结果会保存为 output_dir/视频名/000000.png 这类结构。",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="每个视频抽取的帧数，默认 10。",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="输出图像格式，默认 png。",
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="如果设置该选项，则抽取前 N 帧；否则在整段视频中均匀抽取 N 帧。",
    )
    return parser.parse_args()


def collect_videos(input_dir: Path):
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_frame_indices(total_frames: int, num_frames: int, first_only: bool):
    if total_frames <= 0:
        return []

    if num_frames == 1:
        return [0]

    if first_only:
        return list(range(min(total_frames, num_frames)))

    if total_frames <= num_frames:
        return list(range(total_frames))

    # 均匀采样，确保首尾帧都可能被选中。
    return [round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]


def extract_frames(video_path: Path, output_dir: Path, num_frames: int, image_ext: str, first_only: bool):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = build_frame_indices(total_frames, num_frames, first_only)

    if not indices:
        capture.release()
        raise RuntimeError(f"视频没有可读取的帧: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for save_idx, frame_idx in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok or frame is None:
            print(f"[Warning] 读取失败，跳过 {video_path.name} 的第 {frame_idx} 帧")
            continue

        output_path = output_dir / f"{save_idx:06d}.{image_ext}"
        success = cv2.imwrite(str(output_path), frame)
        if not success:
            print(f"[Warning] 保存失败: {output_path}")

    capture.release()


def main():
    args = parse_args()

    if args.num_frames <= 0:
        raise ValueError("--num-frames 必须大于 0。")

    if not args.input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {args.input_dir}")

    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = collect_videos(args.input_dir)

    if not videos:
        print(f"未在 {args.input_dir} 中找到支持的视频文件。")
        return

    print(f"共找到 {len(videos)} 个视频，开始处理...")
    for video_path in videos:
        video_output_dir = args.output_dir / video_path.stem
        extract_frames(
            video_path=video_path,
            output_dir=video_output_dir,
            num_frames=args.num_frames,
            image_ext=args.image_ext,
            first_only=args.first_only,
        )
        print(f"[Done] {video_path.name} -> {video_output_dir}")

    print("全部处理完成。")


if __name__ == "__main__":
    main()
