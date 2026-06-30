import os
import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser(
        description="Segment raw videos into fixed-length clips and drop short tails.")
    ap.add_argument("--input-dir", default="raw_videos",
                    help="directory of raw videos (default: raw_videos)")
    ap.add_argument("--output-dir", default="processed_videos",
                    help="directory for the output segments (default: processed_videos)")
    ap.add_argument("--segment-time", type=int, default=10,
                    help="segment length in seconds (default: 10)")
    ap.add_argument("--min-duration", type=float, default=4.0,
                    help="discard segments shorter than this many seconds (default: 4.0)")
    args = ap.parse_args()

    input_folder = args.input_dir
    output_folder = args.output_dir
    os.makedirs(output_folder, exist_ok=True)

    print("Starting ffmpeg segmentation...")

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if not os.path.isfile(input_path):
            continue

        print(f"Processing {filename}...")
        name_no_ext = os.path.splitext(filename)[0]
        output_pattern = os.path.join(output_folder, f"{name_no_ext}_part%03d.mp4")

        # split into fixed-length segments, re-encode so keyframes align
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-f", "segment", "-segment_time", str(args.segment_time),
            "-reset_timestamps", "1", output_pattern,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename}: {e.stderr.decode()[:200]}...")

    print(f"Cleaning up short segments (< {args.min_duration}s)...")

    valid_segments = 0
    removed_segments = 0
    for segment_file in os.listdir(output_folder):
        if not segment_file.endswith(".mp4"):
            continue
        segment_path = os.path.join(output_folder, segment_file)

        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            segment_path,
        ]
        try:
            result = subprocess.run(probe_cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            duration_str = result.stdout.strip()
            if duration_str:
                duration = float(duration_str)
                if duration < args.min_duration:
                    os.remove(segment_path)
                    removed_segments += 1
                else:
                    valid_segments += 1
            else:
                os.remove(segment_path)
                removed_segments += 1
        except Exception as e:
            print(f"Error verifying {segment_file}: {e}")

    print("Process complete.")
    print(f"Removed {removed_segments} short segments.")
    print(f"Valid videos remaining: {valid_segments}")


if __name__ == "__main__":
    main()
