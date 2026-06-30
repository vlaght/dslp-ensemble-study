#!/usr/bin/env python3
"""Run the DSLP (phoneme-aligned dual-stream LSTM) component on a video."""
import argparse
import infer


def main():
    ap = argparse.ArgumentParser(description="DSLP (phoneme-aligned dual-stream LSTM) component detector.")
    ap.add_argument("--video", required=True, help="path to the input video")
    args = ap.parse_args()
    prob = infer.predict_dslp(args.video)
    infer.report("DSLP", prob)


if __name__ == "__main__":
    main()
