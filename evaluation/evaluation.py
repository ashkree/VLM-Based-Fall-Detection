"""
evaluation.py — entry point for running fall detection experiments.

Usage:
    python evaluation.py --model "Claude 3.5 Sonnet" --dataset ./dataset/full --output results.csv
"""

import argparse

from dotenv import load_dotenv

from eval_utils import (
    SYSTEM_PROMPT,
    collect_videos,
    print_summary,
    run_evaluation,
    save_csv,
)
from models import get_detector


def main():
    parser = argparse.ArgumentParser(
        description="Run a fall detection evaluation experiment."
    )
    parser.add_argument(
        "--model",
        default="Claude 3.5 Sonnet",
        choices=[
            "GPT-4 Vision",
            "Gemini 1.5 Flash",
            "Claude 3.5 Sonnet",
            "Qwen 2.5 VL",
        ],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset root (must contain FALL/ and NO_FALL/ subfolders)",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to sample per video (ignored by native-video models)",
    )
    args = parser.parse_args()

    # ---- Load model --------------------------------------------------------
    print(f"Loading detector: {args.model}")
    detector = get_detector(args.model)

    # ---- Collect dataset ---------------------------------------------------
    samples = collect_videos(args.dataset)
    if not samples:
        print("No videos found. Check your dataset path.")
        return
    print(f"Found {len(samples)} videos. Starting evaluation...\n")

    # ---- Run experiment ----------------------------------------------------
    rows = run_evaluation(
        detector,
        samples,
        system_prompt=SYSTEM_PROMPT,
        num_frames=args.num_frames,
    )

    # ---- Save & summarise --------------------------------------------------
    save_csv(rows, args.output)
    print_summary(rows)


if __name__ == "__main__":
    load_dotenv()
    main()
