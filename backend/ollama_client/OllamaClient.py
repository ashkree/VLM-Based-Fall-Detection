"""
OllamaClient for Fall Detection using VLMs
Designed for qwen2.5-vl:7b model via Ollama
"""

import base64
import json
import ollama
import requests
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import numpy as np


class OllamaClient:
    """
    Client for interacting with Ollama API for fall detection using VLMs.
    
    Supports single frame and batch frame analysis for temporal fall detection.
    """
    
    def __init__(
        self, 
        host: str = "http://localhost:11434",
        model_name: str = "qwen2.5-vl:7b",
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
        
    def analyze_frame(
        self, 
        image: Union[str, Path, np.ndarray],
        frame_number: int,
        previous_results: Optional[List[Dict[str, Any]]] = None,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single frame for fall detection with temporal context.
        
        Args:
            image: Path to image file or numpy array
            frame_number: Current frame number in the video sequence
            previous_results: List of recent frame analysis results for context (last 1-5)
            custom_prompt: Optional prompt to override default
            
        Returns:
            Dictionary with keys: is_fall, confidence, frame, description
        """
        # TODO: Implement single frame analysis with context
        # 1. Encode the image
        # 2. Build the prompt with previous_results context
        # 3. Make API request
        # 4. Parse JSON response
        # 5. Add frame_number to response
        pass
    
    def analyze_frame_sequence(
        self,
        images: List[Union[str, Path, np.ndarray]],
        window_size: int = 3,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a sequence of frames using sliding window approach.
        
        Args:
            images: List of image paths or numpy arrays
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
    
    def _encode_image(self, image: Union[str, Path, np.ndarray]) -> str:
        """
        Encode image to base64 string for API transmission.
        
        Args:
            image: Path to image file or numpy array
            
        Returns:
            Base64 encoded string
        """
        # TODO: Implement image encoding
        # Handle both file paths and numpy arrays
        pass
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to Ollama API.
        
        Args:
            payload: Request payload containing model, prompt, images, etc.
            
        Returns:
            Parsed response from API
        """
        # TODO: Implement API request
        # 1. Send POST request to self.api_endpoint
        # 2. Handle response/errors
        # 3. Return parsed JSON
        pass
    
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
        
        Returns:
            True if model is available, False otherwise
        """
        # TODO: Implement model verification
        # Hit /api/tags endpoint to list available models
        pass
    
    def health_check(self) -> bool:
        """
        Check if Ollama server is reachable.
        
        Returns:
            True if server is healthy, False otherwise
        """

        return requests.get(self.host) == 200


# Example usage (for reference)
if __name__ == "__main__":
    # Initialize client
    client = OllamaClient(
        host="http://localhost:11434",
        model_name="qwen2.5-vl:7b"
    )
    
    # Check if everything is ready
    if client.health_check() and client.verify_model():
        # Analyze single frame
        result = client.analyze_frame("path/to/frame.jpg")
        print(result)
        
        # Analyze sequence (Phase 2)
        # frames = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
        # result = client.analyze_frame_sequence(frames)
        # print(result)