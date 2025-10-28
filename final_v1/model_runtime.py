# model_runtime.py
# -----------------------------------------------------------------------------
# Minimal integration layer between the Tkinter GUI (app.py) and your model code.
# - Public API:
#     1) init_model() -> ctx
#     2) analyse_video(video_path: str, ctx) -> dict with keys:
#           "class": "FALL" | "NO_FALL" | "UNKNOWN"
#           "desc": str
#           "fall_timestamps": list[float]   # [start_sec, end_sec] or []
#           "metrics": dict                  # optional, may be {}
#
# Design notes:
# - Tries a VLM (Qwen2.5-VL) if available; otherwise uses a motion heuristic.
# - Video backend priority: TorchCodec (if installed) > torchvision > OpenCV.
# - Timestamps are computed from motion spikes so they're consistent and numeric.
# - Keep everything fast and dependency-light; no changes to app.py beyond import.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import io
import math
import json
import re
import warnings
from typing import Any, Dict, List, Tuple, Optional

# -------------------------------
# Optional imports (soft deps)
# -------------------------------
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# TorchCodec (preferred for video if present)
try:
    import torchcodec  # type: ignore
    TORCHCODEC_AVAILABLE = True
except Exception:
    TORCHCODEC_AVAILABLE = False

# torchvision (fallback for video IO)
try:
    from torchvision.io import read_video
    TORCHVISION_AVAILABLE = True
except Exception:
    TORCHVISION_AVAILABLE = False

# OpenCV (last-resort for video IO)
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# Transformers / Qwen2.5-VL
TRANSFORMERS_AVAILABLE = False
QWEN_AVAILABLE = False
ProcessorCls = None
ModelCls = None
try:
    from transformers import AutoProcessor
    ProcessorCls = AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration
    ModelCls = Qwen2_5_VLForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
    QWEN_AVAILABLE = True
except Exception:
    pass

# Pillow for image encoding (to feed frames as images if needed)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration (adjust here if needed)
# -----------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"   # You can swap to a local path
DTYPE = "float16"                             # "float16" or "bfloat16" or "float32"
DEVICE_MAP = "auto"                           # "auto" or explicit device
MAX_FRAMES_FOR_VLM = 8                        # number of key frames to send to VLM
FRAME_SAMPLE_CAP = 256                        # hard cap when reading videos
MOTION_FPS_TARGET = 12                        # downsample fps for motion analysis
MOTION_SPIKE_Z = 2.25                         # z-score threshold to decide a spike
MIN_FALL_DURATION_S = 0.15                    # minimum fall window
DESC_MAX_CHARS = 600                          # keep description compact for GUI


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _sec_fmt(x: float) -> str:
    m, s = divmod(max(0.0, x), 60.0)
    return f"{int(m):02d}:{s:05.2f}"

def _to_gray(frame_bgr_or_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR frame to grayscale float32 in [0,1]."""
    if frame_bgr_or_rgb.ndim != 3 or frame_bgr_or_rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 frame")
    # Try to guess if frame is BGR (OpenCV) or RGB (torchvision/torchcodec).
    # Heuristic: OpenCV path will call this with BGR. We'll assume BGR if OpenCV is used.
    # To keep things robust, allow both and accept minor colour misreads (motion doesn't care).
    # We'll treat array as BGR if it's coming from OpenCV branch.
    return (frame_bgr_or_rgb[..., 0] * 0.114 + frame_bgr_or_rgb[..., 1] * 0.587 + frame_bgr_or_rgb[..., 2] * 0.299) / 255.0

def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    c = np.convolve(x, np.ones(k, dtype=np.float32)/float(k), mode="same")
    return c

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean() if x.size else 0.0
    sd = x.std() if x.size else 1.0
    if sd < 1e-8:
        sd = 1.0
    return (x - mu) / sd

def _clip_desc(text: str, limit: int = DESC_MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit-1] + "…"


# -----------------------------------------------------------------------------
# Video Loading (three backends)
# -----------------------------------------------------------------------------
def _read_with_torchcodec(path: str) -> Tuple[np.ndarray, float]:
    """
    Return (frames_rgb_uint8, fps). torchcodec returns decoded frames as tensors,
    but we'll convert to numpy uint8 RGB for consistency.
    """
    # Minimal torchcodec usage (API may vary; handle safely)
    try:
        reader = torchcodec.VideoReader(path)  # type: ignore[attr-defined]
        meta = reader.metadata
        fps = float(meta.fps) if hasattr(meta, "fps") else 25.0
        frames = []
        count = 0
        for frame in reader:
            if hasattr(frame, "to_rgb"):
                rgb = frame.to_rgb().to_ndarray()  # type: ignore
            else:
                # Fallback: assume ndarray already
                rgb = np.array(frame)
            frames.append(rgb)
            count += 1
            if count >= FRAME_SAMPLE_CAP:
                break
        reader.close()
        if not frames:
            raise RuntimeError("torchcodec decoded zero frames.")
        arr = np.stack(frames, axis=0)
        return arr, fps
    except Exception as e:
        raise RuntimeError(f"torchcodec failed: {e}")

def _read_with_torchvision(path: str) -> Tuple[np.ndarray, float]:
    try:
        vframes, _, info = read_video(path, pts_unit="sec")  # (T, H, W, C) in uint8
        # The new torchvision may return float tensors; handle both
        if hasattr(vframes, "numpy"):
            arr = vframes.numpy()
        else:
            arr = np.asarray(vframes)
        fps = float(info["video_fps"]) if "video_fps" in info else float(info.get("fps", 25.0))
        # Ensure uint8 RGB
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr, fps
    except Exception as e:
        raise RuntimeError(f"torchvision read_video failed: {e}")

def _read_with_opencv(path: str) -> Tuple[np.ndarray, float]:
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV not available.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # OpenCV gives BGR uint8; convert to RGB for consistency
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
        if count >= FRAME_SAMPLE_CAP:
            break
    cap.release()
    if not frames:
        raise RuntimeError("OpenCV decoded zero frames.")
    return np.stack(frames, axis=0), float(fps)

def _read_video_auto(path: str) -> Tuple[np.ndarray, float]:
    last_err = None
    # Prefer TorchCodec (if installed)
    if TORCHCODEC_AVAILABLE:
        try:
            return _read_with_torchcodec(path)
        except Exception as e:
            last_err = e
    # Then torchvision
    if TORCHVISION_AVAILABLE:
        try:
            # Suppress deprecation warning spam
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return _read_with_torchvision(path)
        except Exception as e:
            last_err = e
    # Finally OpenCV
    if OPENCV_AVAILABLE:
        try:
            return _read_with_opencv(path)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No video backend succeeded. Last error: {last_err}")

def _resample_by_fps(frames: np.ndarray, fps: float, target_fps: float) -> Tuple[np.ndarray, float]:
    """Downsample frames to ~target_fps (keeps duration stable; returns new fps)."""
    if target_fps <= 0 or fps <= 0:
        return frames, fps
    if target_fps >= fps:
        return frames, fps
    ratio = fps / target_fps
    idx = (np.arange(0, len(frames)) / ratio).astype(int)
    idx = np.clip(np.unique(idx), 0, len(frames)-1)
    return frames[idx], target_fps


# -----------------------------------------------------------------------------
# Keyframe selection for VLM
# -----------------------------------------------------------------------------
def _select_keyframes(frames: np.ndarray, k: int = MAX_FRAMES_FOR_VLM) -> List[np.ndarray]:
    """
    Pick k frames: a mix of uniform samples and motion-peak frames.
    Ensures uniqueness and keeps chronological order.
    """
    T = len(frames)
    if T == 0:
        return []
    if T <= k:
        return [frames[i] for i in range(T)]

    # Uniform picks
    uni_idx = np.linspace(0, T - 1, num=max(2, k // 2), dtype=int).tolist()

    # Motion peaks
    gray = np.stack([_to_gray(frames[i]) for i in range(T)], axis=0)
    diffs = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))  # (T-1,)
    peak_idx = diffs.argsort()[-max(1, k - len(uni_idx)):] + 1  # +1 to align with next frame
    combined = sorted(set(uni_idx + peak_idx.tolist()))
    if len(combined) > k:
        # Re-uniform to k
        combined = np.linspace(0, len(combined) - 1, num=k, dtype=int).tolist()
    return [frames[i] for i in combined]


# -----------------------------------------------------------------------------
# Motion-based fall window detection (timestamp in seconds)
# -----------------------------------------------------------------------------
def _detect_fall_window(frames: np.ndarray, fps: float) -> List[float]:
    """
    Very fast heuristic:
    - Compute per-step mean absolute difference in grayscale.
    - Smooth & z-score; pick the largest spike above threshold.
    - Expand a small window around the spike to get start/end (seconds).
    Returns [] if no clear spike.
    """
    T = len(frames)
    if T < 3 or fps <= 0:
        return []

    # Downsample for speed/stability
    frames_ds, fps_ds = _resample_by_fps(frames, fps, MOTION_FPS_TARGET)

    gray = np.stack([_to_gray(fr) for fr in frames_ds], axis=0)  # (Td, H, W)
    diffs = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))      # (Td-1,)
    diffs_sm = _moving_average(diffs, max(3, int(fps_ds // 4)))
    z = _zscore(diffs_sm)

    if z.size == 0:
        return []

    peak = int(np.argmax(z))
    if z[peak] < MOTION_SPIKE_Z:
        return []

    # Define a small window around the peak
    pre = int(max(0, peak - max(1, int(fps_ds * 0.2))))
    post = int(min(len(z) - 1, peak + max(1, int(fps_ds * 0.2))))

    # Convert to seconds (use original ds fps)
    start_s = pre / max(1e-6, fps_ds)
    end_s = max(start_s + MIN_FALL_DURATION_S, (post + 1) / max(1e-6, fps_ds))
    return [float(round(start_s, 2)), float(round(end_s, 2))]


# -----------------------------------------------------------------------------
# VLM-based reasoning (optional)
# -----------------------------------------------------------------------------
def _frames_to_pil(frames: List[np.ndarray]) -> List[Image.Image]:
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow not available to convert frames to images.")
    images = []
    for f in frames:
        if f.dtype != np.uint8:
            arr = np.clip(f, 0, 255).astype(np.uint8)
        else:
            arr = f
        images.append(Image.fromarray(arr, mode="RGB"))
    return images

def _vlm_reasoning(images: List[Image.Image], model, processor) -> Dict[str, Any]:
    """
    Ask the VLM for a compact JSON classification and short reasoning.
    We still compute timestamps via motion; the model supplies label + rationale.
    """
    # System prompt keeps output strictly JSON
    system_text = (
        "You are an expert fall-detection assistant. "
        "Look at the provided key frames from a short video. "
        "Classify whether a fall occurred.\n\n"
        "Output ONLY valid JSON with this schema:\n"
        "{\n"
        '  "classification": "FALL" | "NO_FALL",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reasoning": <short explanation up to 300 chars>\n'
        "}\n"
    )

    # Build messages in a Qwen-friendly multimodal chat format
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": "Here are key frames from the video."}] +
                [{"type": "image", "image": img} for img in images]
            ),
        },
    ]

    # Prepare inputs
    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=images, return_tensors="pt")
    else:
        # Fallback: rely on processor to accept (messages) directly
        inputs = processor(messages, return_tensors="pt")

    # Device & dtype dispatch
    if TORCH_AVAILABLE:
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=256,
        do_sample=False,         # deterministic output
        temperature=None
    )
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    if hasattr(processor, "batch_decode"):
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    else:
        # Safe fallback
        text = output_ids[0].tolist() if TORCH_AVAILABLE else str(output_ids)

    # Extract JSON
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {"classification": "UNKNOWN", "confidence": 0.0, "reasoning": "Model did not return JSON."}
    try:
        obj = json.loads(m.group(0))
        cls = str(obj.get("classification", "UNKNOWN")).upper()
        if cls not in ("FALL", "NO_FALL"):
            cls = "UNKNOWN"
        conf = float(obj.get("confidence", 0.0))
        rsn = str(obj.get("reasoning", "")).strip()
        return {"classification": cls, "confidence": max(0.0, min(1.0, conf)), "reasoning": rsn}
    except Exception:
        return {"classification": "UNKNOWN", "confidence": 0.0, "reasoning": "Failed to parse model JSON."}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def init_model() -> Dict[str, Any]:
    """
    One-time initialisation. Returns a ctx dict with whatever we need later.
    Tries to load Qwen2.5-VL; falls back to heuristic-only pipeline if unavailable.
    """
    ctx: Dict[str, Any] = {
        "use_vlm": False,
        "vlm_model": None,
        "vlm_processor": None,
        "video_backend": "auto"
    }

    if TRANSFORMERS_AVAILABLE and QWEN_AVAILABLE and TORCH_AVAILABLE:
        try:
            # Build dtype for torch
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }.get(DTYPE, torch.float16)

            model = ModelCls.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                device_map=DEVICE_MAP,
            )
            processor = ProcessorCls.from_pretrained(MODEL_NAME)
            ctx.update({
                "use_vlm": True,
                "vlm_model": model,
                "vlm_processor": processor,
            })
        except Exception as e:
            warnings.warn(f"[model_runtime] VLM initialisation failed, using heuristic only. Error: {e}")

    # Record which backend we *can* use for video (for logging only)
    if TORCHCODEC_AVAILABLE:
        ctx["video_backend"] = "torchcodec"
    elif TORCHVISION_AVAILABLE:
        ctx["video_backend"] = "torchvision"
    elif OPENCV_AVAILABLE:
        ctx["video_backend"] = "opencv"
    else:
        ctx["video_backend"] = "none"

    return ctx


def analyse_video(video_path: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run analysis on a single video and return GUI-ready dict:
      {
        "class": "FALL" | "NO_FALL" | "UNKNOWN",
        "desc": str,
        "fall_timestamps": [start_s, end_s] | [],
        "metrics": {}
      }
    """
    # ---------- Load video ----------
    if not os.path.isfile(video_path):
        return {
            "class": "UNKNOWN",
            "desc": f"Video not found: {video_path}",
            "fall_timestamps": [],
            "metrics": {}
        }

    try:
        frames, fps = _read_video_auto(video_path)
    except Exception as e:
        return {
            "class": "UNKNOWN",
            "desc": f"Failed to read video: {e}",
            "fall_timestamps": [],
            "metrics": {}
        }

    # ---------- Motion-based timestamps (reliable seconds) ----------
    fall_window = _detect_fall_window(frames, fps)  # [] or [start, end]

    # ---------- Optional VLM reasoning ----------
    label = "UNKNOWN"
    reason = ""
    confidence = None

    if ctx.get("use_vlm") and ctx.get("vlm_model") is not None and ctx.get("vlm_processor") is not None:
        try:
            keyframes = _select_keyframes(frames, k=MAX_FRAMES_FOR_VLM)
            pil_images = _frames_to_pil(keyframes)
            vlm_out = _vlm_reasoning(pil_images, ctx["vlm_model"], ctx["vlm_processor"])
            label = vlm_out.get("classification", "UNKNOWN")
            confidence = vlm_out.get("confidence", None)
            reason = vlm_out.get("reasoning", "")
        except Exception as e:
            # VLM failed; fall back to heuristic only
            label = "UNKNOWN"
            reason = f"VLM reasoning failed; using motion heuristic. ({e})"

    # ---------- Heuristic-only or tie-break ----------
    if label == "UNKNOWN":
        # Decide via motion: strong spike ⇒ FALL, else NO_FALL
        if fall_window:
            label = "FALL"
            if not reason:
                reason = "Detected a sharp motion spike followed by stabilisation."
        else:
            label = "NO_FALL"
            if not reason:
                reason = "No clear motion spike characteristic of a fall."

    # ---------- Compose final description ----------
    if fall_window:
        start_s, end_s = fall_window
        window_txt = f"Estimated fall window: { _sec_fmt(start_s) }–{ _sec_fmt(end_s) }."
    else:
        window_txt = "No fall window detected."

    conf_txt = ""
    if isinstance(confidence, (int, float)):
        conf_txt = f" (confidence: {confidence:.2f})"

    desc = _clip_desc(
        f"{reason}{conf_txt} {window_txt} "
        "Cues: rapid appearance change between frames, followed by reduced motion."
    )

    # ---------- Build result dict ----------
    result = {
        "class": label,
        "desc": desc,
        "fall_timestamps": fall_window if fall_window else [],
        "metrics": {}  # Leave empty; per-video accuracy/precision aren't meaningful
    }
    return result
