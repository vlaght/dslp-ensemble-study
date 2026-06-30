#!/usr/bin/env python3
"""Run the TANoise (temporal-attention BiLSTM on noise-residual features) component on a video."""
import argparse
import infer


def main():
    ap = argparse.ArgumentParser(description="TANoise (temporal-attention BiLSTM on noise-residual features) component detector.")
    ap.add_argument("--video", required=True, help="path to the input video")
    args = ap.parse_args()
    prob = infer.predict_tanoise(args.video)
    infer.report("TANoise", prob)


if __name__ == "__main__":
    main()
