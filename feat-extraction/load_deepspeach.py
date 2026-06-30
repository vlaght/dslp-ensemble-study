import argparse
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser(description="Download DeepSpeak v2 from HuggingFace.")
    ap.add_argument("--dataset", default="faridlab/deepspeak_v2",
                    help="HuggingFace dataset id (default: faridlab/deepspeak_v2)")
    ap.add_argument("--cache-dir", default=None,
                    help="HuggingFace datasets cache directory (default: HF default)")
    args = ap.parse_args()
    load_dataset(args.dataset, trust_remote_code=True, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
