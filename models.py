"""
    This file contains the different models that can be used in the app
"""

import os, json, re, yaml
from abc import ABC, abstractmethod

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
        if 'classification' in result and 'class' not in result:
            result['class'] = result.pop('classification')

        required_keys = ['class', 'confidence', 'reasoning', 'fall_start', 'fall_end']
        for key in required_keys:
            if key not in result:
                result[key] = 0 if 'fall' in key else ''

        return result
    
    def _parse_json(self, str):

        return json.loads(re.sub(r'^```json\n|```$', '', str.strip()))

# ==================== QWEN2.5-7B-VL ===================
class QWEN_2_5_VisionDetector(BaseFallDetector):

    def __init__(self):

        super().__init__("Qwen 2.5 VL")

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from huggingface_hub import snapshot_download
        
        config = self.config["qwen2.5"]

        download_local = config["download_local"]
        local_folder =  config["local_folder"]
        model_type =  config["model"]

        self.max_new_tokens = config["max_new_tokens"]
        self.temperature = config["temperature"]

        if download_local:
            if not os.path.isdir(local_folder):
                snapshot_download(model_name, local_dir=local_folder)
            model_name = local_folder
        else:
            model_name = model_type

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="float16",
            device_map="auto",
            local_files_only=download_local
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            local_files_only=download_local)

    def analyze_video(self, video_path, system_prompt, **kwargs):

        import torch
        from qwen_vl_utils import process_vision_info

        messages = [            
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "Analyse this video and return only JSON."},
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

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return self._parse_json(output_texts[0])

# ==================== GPT-4 VISION ====================
class GPT4VisionDetector(BaseFallDetector):

    def __init__(self, api_key=None):

        from openai import OpenAI

        super().__init__(self.config["gpt"]["name"])
    
        self.model = self.config["gpt"]["model"]

        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))


    def analyze_video(self, video_path, system_prompt, num_frames=8, **kwargs):
        """Analyze video by extracting frames."""
        import base64
        import cv2

        # Extract frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        content = [{"type": "text", "text": system_prompt}]

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize to reduce cost
                height, width = frame.shape[:2]
                if max(height, width) > 1024:
                    scale = 1024 / max(height, width)
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                timestamp = idx / fps if fps > 0 else 0

                content.append({
                    "type": "text",
                    "text": f"\n--- Frame {i + 1}/{num_frames} (Time: {timestamp:.2f}s) ---"
                })

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                        "detail": "high"
                    }
                })

        cap.release()

        content.append({
            "type": "text",
            "text": "\n\nAnalyze this video sequence chronologically to detect falls."
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=1000,
                temperature=0.1
            )

            result_text = response.choices[0].message.content
            result = self._parse_json(result_text)

            # Calculate cost
            image_cost = num_frames * 0.0085
            token_cost = (response.usage.prompt_tokens * 5 +
                          response.usage.completion_tokens * 15) / 1_000_000

            result['api_used'] = self.name
            result['total_cost_estimate'] = {
                'total_usd': round(image_cost + token_cost, 4),
                'images': num_frames,
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            }

        except Exception as e:
            result = {
                "class": "ERROR",
                "confidence": 0.0,
                "reasoning": f"API error: {str(e)}",
                "fall_start": 0,
                "fall_end": 0,
                "api_used": self.name
            }

        return self._standardize_result(result)


# ==================== GOOGLE GEMINI ====================
class GeminiDetector(BaseFallDetector):

    def __init__(self):

        import google.generativeai as genai

        super().__init__(self.config["google"]["name"])
    
        self.model = genai.GenerativeModel(self.config["google"]["model"])
        
        genai.configure(os.getenv('GOOGLE_API_KEY'))

        

    def analyze_video(self, video_path, system_prompt, **kwargs):

        """Analyze video using Gemini (supports native video!)."""
        import google.generativeai as genai

        try:
            # Upload video file
            print("Uploading video to Gemini...")
            video_file = genai.upload_file(path=video_path)

            # Wait for processing
            import time
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError("Video processing failed")

            # Generate content with video
            prompt = f"{system_prompt}\n\nAnalyze this video and return only JSON."
            response = self.model.generate_content(
                [video_file, prompt],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                )
            )

            result_text = response.text
            result = self._parse_json(result_text)

            # Gemini pricing (approximate)
            result['api_used'] = self.name
            result['total_cost_estimate'] = {
                'total_usd': 0.002,  # Very cheap for video
                'note': 'Gemini Flash is very cost-effective'
            }

            # Cleanup
            genai.delete_file(video_file.name)

        except Exception as e:
            result = {
                "class": "ERROR",
                "confidence": 0.0,
                "reasoning": f"Gemini error: {str(e)}",
                "fall_start": 0,
                "fall_end": 0,
                "api_used": self.name
            }

        return self._standardize_result(result)


# ==================== CLAUDE (Anthropic) ====================
class ClaudeDetector(BaseFallDetector):

    def __init__(self):

        from anthropic import Anthropic

        super().__init__(self.config["anthropic"]["name"])
    
        self.model = self.config["anthropic"]["model"]

        self.client = Anthropic(os.getenv('ANTHROPIC_API_KEY'))

    def analyze_video(self, video_path, system_prompt, num_frames=8, **kwargs):
        """Analyze video by extracting frames (Claude doesn't support video)."""
        import base64
        import cv2

        # Extract frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        content = [{"type": "text", "text": system_prompt}]

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize
                height, width = frame.shape[:2]
                if max(height, width) > 1568:  # Claude's max
                    scale = 1568 / max(height, width)
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                timestamp = idx / fps if fps > 0 else 0

                content.append({
                    "type": "text",
                    "text": f"\n--- Frame {i + 1}/{num_frames} (Time: {timestamp:.2f}s) ---"
                })
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame_b64
                    }
                })

        cap.release()

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": content}]
            )

            result_text = response.content[0].text
            result = self._parse_json(result_text)

            # Claude pricing
            image_cost = num_frames * 0.0048  # $4.80 per 1000 images
            token_cost = (response.usage.input_tokens * 3 +
                          response.usage.output_tokens * 15) / 1_000_000

            result['api_used'] = self.name
            result['total_cost_estimate'] = {
                'total_usd': round(image_cost + token_cost, 4),
                'images': num_frames,
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            }

        except Exception as e:
            result = {
                "class": "ERROR",
                "confidence": 0.0,
                "reasoning": f"Claude error: {str(e)}",
                "fall_start": 0,
                "fall_end": 0,
                "api_used": self.name
            }

        return self._standardize_result(result)


# ==================== FACTORY FUNCTION ====================
def get_detector(detector_type, **kwargs):
    """Factory function to get detector instance."""
    detectors = {
        "GPT-4 Vision": GPT4VisionDetector,
        "Gemini 1.5 Flash": GeminiDetector,
        "Claude 3.5 Sonnet": ClaudeDetector,
        "Qwen 2.5 VL": QWEN_2_5_VisionDetector
    }

    if detector_type not in detectors:
        raise ValueError(f"Unknown detector: {detector_type}")

    return detectors[detector_type](**kwargs)