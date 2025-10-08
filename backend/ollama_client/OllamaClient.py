"""
OllamaClient for Fall Detection using VLMs
Designed for qwen2.5-vl:7b model via Ollama
"""

import base64
import json
import ollama
import requests
from typing import Union, List, Dict, Any, Optional, Literal
from pathlib import Path
import numpy as np
import json


class OllamaClient:
    """
    Client for interacting with Ollama API for fall detection using VLMs.
    
    Supports single frame and batch frame analysis for temporal fall detection.
    """
    
    def __init__(
        self, 
        host: str = "http://localhost:11434",
        model_name: str = "qwen2.5vl:7b",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Ollama client.
        
        Args:
            host: Ollama server endpoint
            model_name: Name of the model to use
            system_prompt: Optional system prompt for all requests
        """
        self.host = host
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.api_endpoint = f"{host}/api/generate"

    def analyze_single_frames(
            self, 
            frames: List[str],
            max_memory: int
    ) -> Dict[str, Any]:
        
        """
        Analyze multiple frames for fall detection.
        Supports both single and batch mode

        Args: 
            frames: A list of strings containing path to images
            max_memory: how much of the previous results do we keep

        Returns: 
            Dictionary with keys: 

        """

        previous_results = []
        suspected_fall_frames = []


        """
            Stage 1: Identifying individual frames for falls
        """

        for idx, frame in enumerate(frames):

            print(f"Processing : {idx}/{len(frames)}")

            encoded_frame = self._encode_image(frame)

            result = self.analyze_frame(
                encoded_frame,
                idx,
                previous_results = None if idx == 0 else previous_results
            );

            if result.get("classification") in ["FALL", "POSSIBLE_FALL"]:
                
                result["frame_number"] = idx
                result["encoded_frame"] = encoded_frame
                suspected_fall_frames.append(result)

            previous_results.append(result)

            if len(previous_results) > max_memory:
                previous_results.pop(0)

        """
            Stage 2: Analyzing fall suspected fall frames
        """

        context_info = "\n".join([
            f"Frame {result.get("frame_number")}: {result.get("classification")}"
            f"(confidence: {result.get('confidence', 0):.2f}) - "
            f"{result.get('reasoning', 'N/A')}"
            for result in suspected_fall_frames
        ])

        prompt = f"""
You are given {len(suspected_fall_frames)} frames that were flagged as potential falls during initial screening.

**Flagged frames:**
{context_info}

**Your task:**
Analyze these frames together as a sequence to determine if they truly indicate a fall event.

Give your reasoning in 1 to 3 sentences covering, temporal progression, body position changes, motion patterns, trajectory and environment.

Provide a final determination: Is this a genuine fall event?
"""

        response = ollama.generate(
            model = self.model_name,
            prompt = prompt,
            images = [result.get("encoded_frame") for result in suspected_fall_frames],
            format = {
                "type": "object",
                "properties": {
                    "classification": {
                        "type": "string",
                        "enum": ["FALL", "NO_FALL", "POSSIBLE_FALL"]
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "reasoning": {
                        "type": "string"
                    }
                },
                "required": ["classification", "confidence", "reasoning"]
            }
        )


        return response
        

    def analyze_frame(
        self,
        image: Union[str, Path],
        frame_number: int,
        previous_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        
        # Build context
        context_section = ""
        if previous_results:
            context_section = "\n\nPrevious frame analysis:\n"
            for prev in previous_results[-3:]:
                context_section += f"Frame {prev.get('frame_number', '?')}: "
                context_section += f"{prev.get('classification', 'unknown')}, "
                context_section += f"Confidence={prev.get('confidence', 0):.2f}\n"
        
        message = f"""
    Analyze frame {frame_number} for fall detection.

    {context_section}

    Determine if this frame shows:
    - A fall in progress
    - An imminent fall (loss of balance, slipping)
    - Post-fall state (person on ground)
    - Normal activity (no fall)

    Use the temporal context from previous frames if available.
    Always respond in JSON
    """
        
        response = ollama.generate(
            model=self.model_name,
            prompt=message,
            images=[image],
            format={
                "type": "object",
                "properties": {
                    "classification": {
                        "type": "string",
                        "enum": ["FALL", "NO_FALL", "POSSIBLE_FALL"]
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "reasoning": {
                        "type": "string"
                    }
                },
                "required": ["classification", "confidence", "reasoning"]
            }
        )
        
        result = json.loads(response.response)
        return result
        


    def analyze_frame_sequence(
        self,
        frames: List[str],
        window_size: int = 3,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a sequence of frames using sliding window approach.
        
        Args:
            images: List of image paths
            window_size: Number of frames to analyze together
            custom_prompt: Optional prompt to override default
            
        Returns:
            Dictionary with keys: is_fall, confidence, frames, description
        """
        # TODO: Implement batch frame analysis (Phase 2)
        # 1. Create sliding windows
        # 2. Process each window
        # 3. Aggregate results
        pass
    
    def _encode_image(self, image: Union[str, Path]) -> str:
        """
        Encode image to base64 string for API transmission.
        
        Args:
            image: Path to image file
            
        Returns:
            Base64 encoded string
        """

        image_path = Path(image)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not image_path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        with open(image_path, 'rb') as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        
        return encoded
    
    def _build_prompt(self, custom_prompt: Optional[str] = None) -> str:
        """
        Build the prompt for fall detection analysis.
        
        Args:
            custom_prompt: Optional custom prompt to use instead of default
            
        Returns:
            Complete prompt string
        """
        # TODO: Implement prompt building
        # Consider system_prompt + task-specific prompt
        # Ensure it requests JSON format response
        pass
    
    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse and validate the JSON response from the model.
        
        Args:
            raw_response: Raw text response from model
            
        Returns:
            Validated dictionary with required keys
        """
        # TODO: Implement response parsing
        # 1. Extract JSON from response
        # 2. Validate required fields
        # 3. Handle malformed responses
        pass
    
    def verify_model(self) -> bool:
        """
        Verify that the specified model is available in Ollama.
        If not available, attempt to pull it.
        
        Returns:
            True if model is available or successfully pulled, False otherwise
        """
        try:
            # Check if Ollama server is running
            print(f"Checking for model: {self.model_name}")
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            
            if response.status_code != 200:
                print(f"✗ Ollama server returned status code: {response.status_code}")
                return False
            
            # Parse available models
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            # Check if model exists
            if self.model_name in available_models:
                print(f"✓ Model '{self.model_name}' is available")
                return True
            
            # Model not found - attempt to pull it
            print(f"✗ Model '{self.model_name}' not found")
            print(f"  Available models: {', '.join(available_models) if available_models else 'None'}")
            print(f"\nAttempting to download model '{self.model_name}'...")
            print("⚠️  This may take several minutes depending on your internet connection")
            
            try:
                # Pull the model
                ollama.pull(self.model_name)
                print(f"✓ Successfully downloaded '{self.model_name}'")
                
                # Verify the model is now available
                response = requests.get(f"{self.host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    
                    if self.model_name in available_models:
                        print(f"✓ Model '{self.model_name}' verified and ready")
                        return True
                
                print(f"✗ Model pull completed but model not found in list")
                return False
                
            except Exception as pull_error:
                print(f"✗ Failed to download model: {pull_error}")
                print(f"\n  You can manually download it by running:")
                print(f"    ollama pull {self.model_name}")
                return False
        
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to Ollama server")
            print(f"  Please ensure Ollama is running on {self.host}")
            print("\n  Start Ollama with: ollama serve")
            return False
        
        except requests.exceptions.Timeout:
            print(f"✗ Connection to Ollama timed out")
            return False
        
        except (KeyError, ValueError) as e:
            print(f"✗ Error parsing model list: {e}")
            return False
        
        except Exception as e:
            print(f"✗ Unexpected error: {type(e).__name__}: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if Ollama server is reachable and optionally verify model availability.
        
        Args:
            check_model: Model name to verify (e.g., 'qwen2.5:7b-instruct'). 
                        If None, only checks server connectivity.
        
        Returns:
            True if server is healthy (and model is available if specified), False otherwise
        """
        try:
            # Check if Ollama server is running
            response = requests.get(self.host, timeout=5)
            
            if response.status_code != 200:
                print(f"✗ Ollama server returned status code: {response.status_code}")
                return False
            
            print("✓ Ollama server is running")

            return True
                
            
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to Ollama server")
            print(f"  Connection refused on {self.host}")
            print("\n📋 To fix this:")
            print("  1. Start Ollama:")
            print("     • Linux/macOS: ollama serve")
            print("     • Or use: systemctl start ollama")
            print("     • macOS app: Open Ollama from Applications")
            print("  2. Verify it's running:")
            print(f"     curl {self.host}/api/tags")
            return False
            
        except requests.exceptions.Timeout:
            print(f"✗ Connection to Ollama timed out (>{5}s)")
            print("  The server may be overloaded or unresponsive")
            return False
            
        except Exception as e:
            print(f"✗ Unexpected error checking Ollama health:")
            print(f"  {type(e).__name__}: {e}")
            return False
    
# Example usage (for reference)
if __name__ == "__main__":
    # Initialize client
    client = OllamaClient(
        host="http://localhost:11434",
        model_name="qwen2.5-vl:7b"
    )

    client.health_check()
