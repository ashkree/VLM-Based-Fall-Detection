import os
import json
import re
from abc import ABC, abstractmethod


# Base class for all detectors
class BaseFallDetector(ABC):
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


# ==================== GPT-4 VISION ====================
class GPT4VisionDetector(BaseFallDetector):
    def __init__(self, api_key=None):
        super().__init__("GPT-4 Vision")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o"

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
            result_text = re.sub(r'^```json\n|```$', '', result_text.strip())
            result = json.loads(result_text)

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
    def __init__(self, api_key=None):
        super().__init__("Gemini 1.5 Flash")
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')

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
            result_text = re.sub(r'^```json\n|```$', '', result_text.strip())
            result = json.loads(result_text)

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
    def __init__(self, api_key=None):
        super().__init__("Claude 3.5 Sonnet")
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))
        self.model = "claude-3-5-sonnet-20241022"

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
            result_text = re.sub(r'^```json\n|```$', '', result_text.strip())
            result = json.loads(result_text)

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
    }

    if detector_type not in detectors:
        raise ValueError(f"Unknown detector: {detector_type}")

    return detectors[detector_type](**kwargs)