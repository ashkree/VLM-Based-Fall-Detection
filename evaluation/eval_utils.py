"""
eval_utils.py — reusable evaluation helpers for fall detection experiments.

Functions
---------
collect_videos(dataset_root)    Scan dataset folders and return labelled samples.
eval_video(detector, sample, system_prompt, num_frames)
                                Run inference on a single video and return a result row.
run_evaluation(detector, samples, system_prompt, num_frames)
                                Run eval_video over a full sample list and return all rows.
save_csv(rows, output_csv)      Write result rows to a CSV file.
print_summary(rows)             Print accuracy / error summary to stdout.
"""

import csv
import os
import traceback

LABELS = ["FALL", "NO_FALL"]

SYSTEM_PROMPT = (
    "You are a fall detection expert. Analyze the provided video frames and return ONLY "
    "a JSON object with the following keys:\n"
    '  "class": "FALL" or "NO_FALL",\n'
    '  "confidence": float between 0 and 1,\n'
    '  "reasoning": brief explanation,\n'
    '  "fall_start": timestamp in seconds (0 if no fall),\n'
    '  "fall_end": timestamp in seconds (0 if no fall)\n'
    "Return only valid JSON, no extra text."
)

CSV_FIELDS = [
    "filename",
    "actual_label",
    "predicted_label",
    "confidence",
    "video_length_s",
    "inference_time_s",
    "input_tokens",
    "output_tokens",
]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def collect_videos(dataset_root: str) -> list[dict]:
    """
    Scan a dataset directory for labelled MP4 videos.

    Expected structure:
        dataset_root/
            FALL/
                video1.mp4 ...
            NO_FALL/
                video2.mp4 ...

    Returns:
        List of dicts with keys: 'path', 'filename', 'actual_label'.
    """
    samples = []
    for label in LABELS:
        folder = os.path.join(dataset_root, label)
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder not found, skipping: {folder}")
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(".mp4"):
                samples.append(
                    {
                        "path": os.path.join(folder, fname),
                        "filename": fname,
                        "actual_label": label,
                    }
                )
    return samples


# ---------------------------------------------------------------------------
# Per-video evaluation
# ---------------------------------------------------------------------------


def eval_video(
    detector, sample: dict, system_prompt: str = SYSTEM_PROMPT, num_frames: int = 8
) -> dict:
    """
    Run inference on a single video and return a result row dict.

    Args:
        detector:      An initialised BaseFallDetector instance.
        sample:        Dict with keys 'path', 'filename', 'actual_label'
                       (as returned by collect_videos).
        system_prompt: Prompt passed to the model.
        num_frames:    Number of frames to sample (ignored by native-video models).

    Returns:
        Dict with all CSV_FIELDS populated. On failure, predicted_label is set
        to "ERROR" and the exception message is stored in an 'error' key.
    """
    path = sample["path"]
    fname = sample["filename"]
    actual = sample["actual_label"]

    duration = detector.get_video_duration(path)

    row = {
        "filename": fname,
        "actual_label": actual,
        "predicted_label": "ERROR",
        "confidence": "",
        "video_length_s": duration,
        "inference_time_s": "",
        "input_tokens": "",
        "output_tokens": "",
        "error": "",
    }

    try:
        result = detector.analyze_video(path, system_prompt, num_frames=num_frames)
        row["predicted_label"] = str(result.get("class", "ERROR")).upper()
        row["confidence"] = result.get("confidence", "")
        row["inference_time_s"] = result.get("inference_time_s", "")
        row["input_tokens"] = result.get("input_tokens", "")
        row["output_tokens"] = result.get("output_tokens", "")
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        print(f"\n[ERROR] {fname}: {row['error']}")
        traceback.print_exc()

    return row


# ---------------------------------------------------------------------------
# Full dataset evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    detector,
    samples: list[dict],
    system_prompt: str = SYSTEM_PROMPT,
    num_frames: int = 8,
) -> list[dict]:
    """
    Run eval_video over every sample and return all result rows.

    Args:
        detector:      An initialised BaseFallDetector instance.
        samples:       List of sample dicts from collect_videos().
        system_prompt: Prompt passed to the model.
        num_frames:    Number of frames to sample per video.

    Returns:
        List of result row dicts (one per video).
    """
    rows = []
    total = len(samples)

    for i, sample in enumerate(samples, 1):
        fname = sample["filename"]
        actual = sample["actual_label"]

        print(f"[{i}/{total}] {actual}/{fname} ...", end=" ", flush=True)

        row = eval_video(detector, sample, system_prompt, num_frames)

        status = row["predicted_label"]
        if row["error"]:
            print(f"ERROR ({row['error']})")
        else:
            correct = "✓" if status == actual else "✗"
            print(f"{correct} {status}  ({row['inference_time_s']}s)")

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_csv(rows: list[dict], output_csv: str) -> None:
    """Write result rows to a CSV file (error column excluded)."""
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {output_csv}")


def print_summary(rows: list[dict]) -> None:
    """Print a short accuracy / error summary to stdout."""
    total = len(rows)
    errors = sum(1 for r in rows if r["predicted_label"] == "ERROR")
    valid = total - errors
    correct = sum(1 for r in rows if r["actual_label"] == r["predicted_label"])

    tp = sum(
        1
        for r in rows
        if r["actual_label"] == "FALL" and r["predicted_label"] == "FALL"
    )
    fp = sum(
        1
        for r in rows
        if r["actual_label"] == "NO_FALL" and r["predicted_label"] == "FALL"
    )
    fn = sum(
        1
        for r in rows
        if r["actual_label"] == "FALL" and r["predicted_label"] == "NO_FALL"
    )

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    accuracy_str = f"{100 * correct / valid:.1f}%" if valid else "N/A"

    print("\n" + "=" * 40)
    print(f"  Total videos : {total}")
    print(f"  Errors       : {errors}")
    print(f"  Accuracy     : {correct}/{valid} ({accuracy_str})")
    print(f"  Precision    : {precision:.3f}")
    print(f"  Recall       : {recall:.3f}")
    print(f"  F1 Score     : {f1:.3f}")
    print("=" * 40)
