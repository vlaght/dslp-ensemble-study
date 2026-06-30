#!/usr/bin/env python3
"""Run the TAFreq (temporal-attention BiLSTM on frequency features) component on a video."""
import argparse
import infer


def main():
    ap = argparse.ArgumentParser(description="TAFreq (temporal-attention BiLSTM on frequency features) component detector.")
    ap.add_argument("--video", required=True, help="path to the input video")
    args = ap.parse_args()
    prob = infer.predict_tafreq(args.video)
    infer.report("TAFreq", prob)


if __name__ == "__main__":
    main()
