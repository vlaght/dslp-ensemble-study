#!/usr/bin/env python3
"""Run the final detector (DSLP + TAFreq + TANoise, unweighted-mean ensemble) on a video."""
import argparse
import infer


def main():
    ap = argparse.ArgumentParser(description="Final ensemble face-forgery detector.")
    ap.add_argument("--video", required=True, help="path to the input video")
    args = ap.parse_args()
    p = infer.predict_ensemble(args.video)
    print("--- components ---")
    infer.report("DSLP   ", p["dslp"])
    infer.report("TAFreq ", p["tafreq"])
    infer.report("TANoise", p["tanoise"])
    print("--- ensemble ---")
    infer.report("ENSEMBLE", p["ensemble"])


if __name__ == "__main__":
    main()
