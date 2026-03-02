"""
This file contains the different models that can be used in the app
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod

import yaml

# ==================== BASE CLASS ===================


class BaseFallDetector(ABC):
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def analyze_video(self, video_path, system_prompt, **kwargs):
        """Analyze video and return standardized JSON result."""
        pass

    def _standardize_result(self, result):
        """Ensure result has consistent format."""
        # Map 'classification' to 'class' if needed
        if "classification" in result and "class" not in result:
            result["class"] = result.pop("classification")

        required_keys = ["class", "confidence", "reasoning", "fall_start", "fall_end"]
        for key in required_keys:
            if key not in result:
                result[key] = 0 if "fall" in key else ""

        # Ensure inference metadata keys always exist
        result.setdefault("inference_time_s", None)
        result.setdefault("input_tokens", None)
        result.setdefault("output_tokens", None)

        return result

    def _parse_json(self, s):
        return json.loads(re.sub(r"^```json\n|```$", "", s.strip()))

    def get_video_duration(self, video_path: str) -> float:
        """Return video duration in seconds."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return round(frame_count / fps, 2) if fps > 0 else 0.0

    def _extract_frames(
        self, video_path: str, num_frames: int, max_dimension: int
    ) -> list:
        """
        Extract evenly-spaced frames from a video.

        Args:
            video_path:    Path to the source video file.
            num_frames:    Number of frames to sample.
            max_dimension: Longest edge will be scaled down to this pixel limit.

        Returns:
            List of dicts with keys: 'index', 'num_frames', 'timestamp', 'frame_b64'.
            Subclasses format these into their own API-specific content blocks.
        """
        import base64

        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        frames = []
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            height, width = frame.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(
                {
                    "index": i,
                    "num_frames": num_frames,
                    "timestamp": idx / fps if fps > 0 else 0,
                    "frame_b64": base64.b64encode(buffer).decode("utf-8"),
                }
            )

        cap.release()
        return frames


# ==================== QWEN2.5-7B-VL ===================


class QWEN_2_5_VisionDetector(BaseFallDetector):
    def __init__(self):
        super().__init__("Qwen 2.5 VL")

        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        config = self.config["qwen2.5"]

        download_local = config["download_local"]
        local_folder = config["local_folder"]
        model_type = config["model"]

        self.max_new_tokens = config["max_new_tokens"]
        self.temperature = config["temperature"]

        if download_local:
            if not os.path.isdir(local_folder):
                snapshot_download(model_type, local_dir=local_folder)
            model_name = local_folder
        else:
            model_name = model_type

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="float16",
            device_map="auto",
            local_files_only=download_local,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=download_local
        )

    def analyze_video(self, video_path, system_prompt, **kwargs):
        import torch
        from qwen_vl_utils import process_vision_info

        # Qwen supports native video input, so no frame extraction needed
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {
                        "type": "text",
                        "text": "Analyse this video and return only JSON.",
                    },
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        input_token_count = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
            )
        inference_time = round(time.perf_counter() - t0, 3)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_token_count = sum(len(ids) for ids in generated_ids_trimmed)

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        result = self._parse_json(output_texts[0])
        result["inference_time_s"] = inference_time
        result["input_tokens"] = input_token_count
        result["output_tokens"] = output_token_count
        return result


# ==================== GPT-4 VISION ====================


class GPT4VisionDetector(BaseFallDetector):
    def __init__(self, api_key=None):
        from openai import OpenAI

        super().__init__(self.config["gpt"]["name"])

        self.model = self.config["gpt"]["model"]
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def analyze_video(self, video_path, system_prompt, num_frames=8, **kwargs):
        """Analyze video by extracting frames."""
        frames = self._extract_frames(video_path, num_frames, max_dimension=1024)

        content = [{"type": "text", "text": system_prompt}]
        for f in frames:
            content.append(
                {
                    "type": "text",
                    "text": f"\n--- Frame {f['index'] + 1}/{f['num_frames']} (Time: {f['timestamp']:.2f}s) ---",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{f['frame_b64']}",
                        "detail": "high",
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": "\n\nAnalyze this video sequence chronologically to detect falls.",
            }
        )

        try:
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=1000,
                temperature=0.1,
            )
            inference_time = round(time.perf_counter() - t0, 3)

            result = self._parse_json(response.choices[0].message.content)

            image_cost = num_frames * 0.0085
            token_cost = (
                response.usage.prompt_tokens * 5 + response.usage.completion_tokens * 15
            ) / 1_000_000

            result["api_used"] = self.name
            result["inference_time_s"] = inference_time
            result["input_tokens"] = response.usage.prompt_tokens
            result["output_tokens"] = response.usage.completion_tokens
            result["total_cost_estimate"] = {
                "total_usd": round(image_cost + token_cost, 4),
                "images": num_frames,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        except Exception as e:
            result = {
                "class": "ERROR",
                "confidence": 0.0,
                "reasoning": f"API error: {str(e)}",
                "fall_start": 0,
                "fall_end": 0,
                "api_used": self.name,
            }

        return self._standardize_result(result)


# ==================== GOOGLE GEMINI ====================


class GeminiDetector(BaseFallDetector):
    def __init__(self):
        import google.generativeai as genai

        super().__init__(self.config["google"]["name"])

        self.model = genai.GenerativeModel(self.config["google"]["model"])
        genai.configure(os.getenv("GOOGLE_API_KEY"))

    def analyze_video(self, video_path, system_prompt, **kwargs):
        """Analyze video using Gemini (supports native video)."""
        import google.generativeai as genai

        try:
            print("Uploading video to Gemini...")
            video_file = genai.upload_file(path=video_path)

            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError("Video processing failed")

            prompt = f"{system_prompt}\n\nAnalyze this video and return only JSON."
            t0 = time.perf_counter()
            response = self.model.generate_content(
                [video_file, prompt],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                ),
            )
            inference_time = round(time.perf_counter() - t0, 3)

            result = self._parse_json(response.text)
            result["api_used"] = self.name
            result["inference_time_s"] = inference_time
            result["input_tokens"] = response.usage_metadata.prompt_token_count
            result["output_tokens"] = response.usage_metadata.candidates_token_count
            result["total_cost_estimate"] = {
                "total_usd": 0.002,
                "note": "Gemini Flash is very cost-effective",
            }

            genai.delete_file(video_file.name)

        except Exception as e:
            result = {
                "class": "ERROR",
                "confidence": 0.0,
                "reasoning": f"Gemini error: {str(e)}",
                "fall_start": 0,
                "fall_end": 0,
                "api_used": self.name,
            }

        return self._standardize_result(result)


# ==================== CLAUDE (Anthropic) ====================


class ClaudeDetector(BaseFallDetector):
    def __init__(self):
        from anthropic import Anthropic

        super().__init__(self.config["anthropic"]["name"])

        self.model = self.config["anthropic"]["model"]
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def analyze_video(self, video_path, system_prompt, num_frames=8, **kwargs):
        """Analyze video by extracting frames (Claude doesn't support native video)."""
        frames = self._extract_frames(video_path, num_frames, max_dimension=1568)

        content = [{"type": "text", "text": system_prompt}]
        for f in frames:
            content.append(
                {
                    "type": "text",
                    "text": f"\n--- Frame {f['index'] + 1}/{f['num_frames']} (Time: {f['timestamp']:.2f}s) ---",
                }
            )
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": f["frame_b64"],
                    },
                }
            )

        try:
            t0 = time.perf_counter()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": content}],
            )
            inference_time = round(time.perf_counter() - t0, 3)

            result = self._parse_json(response.content[0].text)

            image_cost = num_frames * 0.0048
            token_cost = (
                response.usage.input_tokens * 3 + response.usage.output_tokens * 15
            ) / 1_000_000

            result["api_used"] = self.name
            result["inference_time_s"] = inference_time
            result["input_tokens"] = response.usage.input_tokens
            result["output_tokens"] = response.usage.output_tokens
            result["total_cost_estimate"] = {
                "total_usd": round(image_cost + token_cost, 4),
                "images": num_frames,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        except Exception as e:
            result = {
                "class": "ERROR",
                "confidence": 0.0,
                "reasoning": f"Claude error: {str(e)}",
                "fall_start": 0,
                "fall_end": 0,
                "api_used": self.name,
            }

        return self._standardize_result(result)


# ==================== FACTORY FUNCTION ====================


def get_detector(detector_type, **kwargs):
    """Factory function to get detector instance."""
    detectors = {
        "GPT-4 Vision": GPT4VisionDetector,
        "Gemini 1.5 Flash": GeminiDetector,
        "Claude 3.5 Sonnet": ClaudeDetector,
        "Qwen 2.5 VL": QWEN_2_5_VisionDetector,
    }

    if detector_type not in detectors:
        raise ValueError(f"Unknown detector: {detector_type}")

    return detectors[detector_type](**kwargs)
