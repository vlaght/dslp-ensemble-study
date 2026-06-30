import os
import argparse
import yt_dlp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(
        description="Download the augmented YouTube source clips listed in a URL file.")
    ap.add_argument("--urls", default=os.path.join(SCRIPT_DIR, "videos.txt"),
                    help="path to a text file with one YouTube URL per line "
                         "(default: videos.txt next to this script)")
    ap.add_argument("--output-dir", default="raw_videos",
                    help="directory to save downloaded videos (default: raw_videos)")
    args = ap.parse_args()

    with open(args.urls, "r") as videos_txt:
        video_urls = [line.strip() for line in videos_txt if line.strip()]

    output_folder = args.output_dir
    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        "outtmpl": os.path.join(output_folder, "%(id)s.%(ext)s"),
        "format": "bv+ba/best",
        "quiet": False,
        "no_warnings": True,
        "ignoreerrors": True,  # Continue even if some videos are unavailable
    }

    print(f"Reading {len(video_urls)} URLs from {args.urls}")
    print("Checking and downloading videos...")

    # Existing video IDs (filename without extension) to skip re-downloads
    existing_ids = set()
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            if os.path.isfile(os.path.join(output_folder, f)):
                file_id, _ = os.path.splitext(f)
                existing_ids.add(file_id)

    stats = {"downloaded": 0, "existed": 0, "error": 0}

    with (
        yt_dlp.YoutubeDL({"quiet": True, "ignoreerrors": True}) as ydl_extract,
        yt_dlp.YoutubeDL(ydl_opts) as ydl_download,
    ):
        for url in video_urls:
            video_id = None
            try:
                info = ydl_extract.extract_info(url, download=False)
                if info:
                    video_id = info.get("id")
            except Exception:
                pass

            if video_id and video_id in existing_ids:
                print(f"Skipping {url} (ID: {video_id}) - already downloaded.")
                stats["existed"] += 1
                continue

            print(f"Processing {url}...")
            try:
                ret_code = ydl_download.download([url])
                if ret_code == 0:
                    stats["downloaded"] += 1
                else:
                    stats["error"] += 1
            except Exception as e:
                print(f"Error processing {url}: {e}")
                stats["error"] += 1

    downloaded_files = os.listdir(output_folder)
    print(f"\nDownloaded files in '{output_folder}': {downloaded_files}")

    print("\n--- Download Statistics ---")
    print(f"Downloaded: {stats['downloaded']} videos")
    print(f"Existed: {stats['existed']} videos")
    print(f"Error: {stats['error']} videos")


if __name__ == "__main__":
    main()
