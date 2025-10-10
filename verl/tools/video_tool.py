# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import torch
import torchcodec.decoders
import torchvision.transforms.functional as TF
from PIL import Image

from verl.utils.reward_score import video_holmes
from verl.utils.rollout_trace import rollout_trace_op
import time

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def encode_timestamp_to_str(timestamp: int) -> str:
    return f"{datetime.fromtimestamp(timestamp, tz=timezone.utc):%H:%M:%S}"

def decode_timestamp_to_str(timestamp_str: str) -> int:
    t = datetime.strptime(timestamp_str, "%H:%M:%S").time()
    return t.hour * 3600 + t.minute * 60 + t.second


def torch_frames_to_pil_images(frames: torch.Tensor, size: Optional[int] = None) -> list[Image.Image]:
    """Convert torch tensor frames to PIL Images with optional resizing.

    Args:
        frames: Torch tensor of shape (T, C, H, W) where T is number of frames
        size: Optional target size for the shorter side. If None, no resizing is done.

    Returns:
        List of PIL Image objects
    """
    pil_images = []

    for i in range(frames.shape[0]):
        frame = frames[i]  # Shape: (C, H, W)

        # Convert to numpy: (C, H, W) -> (H, W, C)
        frame_np = frame.permute(1, 2, 0).cpu().numpy()

        # Convert to uint8 (frames from torchcodec are typically uint8 already)
        if frame_np.dtype != np.uint8:
            # If float, assume [0, 1] range and scale to [0, 255]
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

        # Create PIL Image
        pil_image = Image.fromarray(frame_np)

        # Resize if size is specified
        if size is not None:
            pil_image = TF.resize(pil_image, size)

        pil_images.append(pil_image)

    return pil_images


class ExtractFramesTool(BaseTool):
    """A tool for uniformly extracting frames from a video.

    This tool extracts frames uniformly distributed between start and end timestamps
    from a video file, with optional resizing.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory with video_path, num_frames, size
        execute: Execute frame extraction based on start/end timestamps
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ExtractFramesTool.

        Tool schema example:
        {
            "type": "function",
            "function": {
                "name": "extract_frames",
                "description": "A tool for uniformly extracting frames from a video...",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "description": "Start timestamp in HH:MM:SS format"},
                        "end": {"type": "string", "description": "End timestamp in HH:MM:SS format"}
                    },
                    "required": ["start", "end"]
                }
            }
        }

        Args:
            config: Tool configuration dict
            tool_schema: OpenAI function tool schema
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        logger.info(f"Initialized ExtractFramesTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance for video frame extraction.

        Args:
            instance_id: Optional unique identifier for the instance
            **kwargs: Should contain 'create_kwargs' dict with:
                - video_path: Path to the video file (required)
                - num_frames: Number of frames to extract uniformly between start and end (default: 4)
                - size: Target size for the shorter side of extracted frames (default: 256)

        Returns:
            Tuple of (instance_id, ToolResponse)

        Raises:
            ValueError: If video_path is missing or video file doesn't exist
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Extract create_kwargs from kwargs
        create_kwargs = kwargs.get("create_kwargs", {})
        
        # Extract parameters from create_kwargs with defaults
        video_path = create_kwargs.get("video_path")
        num_frames = create_kwargs.get("num_frames", 4)
        size = create_kwargs.get("size", 256)

        if video_path is None:
            raise ValueError("Missing required 'video_path' parameter in create_kwargs")

        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        # Load video to get metadata
        try:
            vr = torchcodec.decoders.VideoDecoder(video_path)
            duration = vr.metadata.duration_seconds
            duration = np.floor(duration)
        except Exception as e:
            raise ValueError(f"Failed to load video from {video_path}: {e}") from e

        self._instance_dict[instance_id] = {
            "size": size,
            "duration": duration,
            "num_frames": num_frames,
            "decoder": vr,
        }

        logger.info(f"Created ExtractFramesTool instance {instance_id} for video: {video_path} (duration: {duration}s)")

        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute frame extraction from video.

        Args:
            instance_id: The instance id of the tool
            parameters: Dict containing 'start' and 'end' timestamp strings in HH:MM:SS format
            **kwargs: Additional execution kwargs

        Returns:
            Tuple of (ToolResponse with extracted frames, reward, metrics)
        """
        start_str = parameters.get("start")
        end_str = parameters.get("end")

        if not start_str or not end_str:
            error_msg = "Error: Both 'start' and 'end' parameters are required in HH:MM:SS format."
            logger.warning(f"Tool execution failed: {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        instance_data = self._instance_dict[instance_id]
        num_frames = instance_data["num_frames"]
        size = instance_data["size"]
        duration = instance_data["duration"]
        duration_str = encode_timestamp_to_str(duration)
        num_frames = instance_data["num_frames"]
        vr = instance_data["decoder"]

        try:
            # Parse timestamps
            start_sec = decode_timestamp_to_str(start_str)
            end_sec = decode_timestamp_to_str(end_str)

            max_sec = vr.metadata.end_stream_seconds - 1e-6
            min_sec = vr.metadata.begin_stream_seconds

            start_sec = np.clip(start_sec, min_sec, max_sec).item()
            end_sec = np.clip(end_sec, min_sec, max_sec).item()

            # Validate timestamps
            if start_sec < 0 or end_sec > duration:
                error_msg = (
                    f"Error: Timestamps out of range. Video duration is {duration_str}, "
                    f"but requested start={start_str}, end={end_str}"
                )
                logger.warning(error_msg)
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            if start_sec >= end_sec:
                error_msg = f"Error: Start time ({start_str}) must be before end time ({end_str})"
                logger.warning(error_msg)
                return ToolResponse(text=error_msg), -0.05, {"success": False}
            
            # Generate uniform timestamps between start and end
            timestamps = np.round(np.linspace(start_sec, end_sec, num_frames), 0).astype(int)
            # Remove almost overlapping timestamps
            timestamps = np.array(sorted(list(set(timestamps.tolist()))))
            timestamp_strs = [encode_timestamp_to_str(t) for t in timestamps]

            timestamps = np.clip(timestamps, min_sec, max_sec) # Safe to be used for indexing


            # Extract frames at these timestamps
            try:
                frames = vr.get_frames_played_at(timestamps.tolist()).data # Returns torch tensor: T x C x H x W
            except Exception as e:
                print(f"Error extracting frames: {start_sec} {end_sec}")
                raise e

            logger.info(
                f"Extracted {frames.shape[0]} frames from {start_str} to {end_str} (duration: {duration_str}), "
                f"shape: {frames.shape}, dtype: {frames.dtype}"
            )

            # Convert to PIL Images with resizing
            pil_images = torch_frames_to_pil_images(frames, size=size)

            response_text = (
                f"Extracted {len(pil_images)} frames uniformly from {start_str} to {end_str} (duration: {duration_str}). "
                f"Frames with timestamps: {", ".join([f'{t} <image>' for t in timestamp_strs])}"
            )

            return (
                ToolResponse(image=pil_images, text=response_text),
                0.0,
                {"success": True, "num_frames": len(pil_images)},
            )

        except Exception as e:
            error_msg = f"Error extracting frames: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResponse(text=error_msg), -0.05, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance and cleanup resources.

        Args:
            instance_id: The instance id of the tool
        """
        if instance_id in self._instance_dict:
            # Clean up decoder if needed
            instance_data = self._instance_dict[instance_id]
            if "decoder" in instance_data:
                del instance_data["decoder"]
            del self._instance_dict[instance_id]
            logger.info(f"Released ExtractFramesTool instance {instance_id}")


class ZoomInFrameTool(BaseTool):
    """A tool for zooming in on a specific frame from a video.

    This tool extracts a single frame at a given timestamp and optionally resizes it
    to a higher resolution for detailed inspection.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance with video_path and size
        execute: Execute frame extraction at a specific timestamp
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ZoomInFrameTool.

        Tool schema example:
        {
            "type": "function",
            "function": {
                "name": "zoom_in_frame",
                "description": "A tool for zooming in on a specific frame from the video",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string", "description": "Timestamp in HH:MM:SS format"}
                    },
                    "required": ["timestamp"]
                }
            }
        }

        Args:
            config: Tool configuration dict
            tool_schema: OpenAI function tool schema
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        logger.info(f"Initialized ZoomInFrameTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance for frame zoom.

        Args:
            instance_id: Optional unique identifier for the instance
            **kwargs: Should contain 'create_kwargs' dict with:
                - video_path: Path to the video file (required)
                - size: Target size for the shorter side of the extracted frame (default: 512)

        Returns:
            Tuple of (instance_id, ToolResponse)

        Raises:
            ValueError: If video_path is missing or video file doesn't exist
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Extract create_kwargs from kwargs
        create_kwargs = kwargs.get("create_kwargs", {})
        
        # Extract parameters from create_kwargs with defaults
        video_path = create_kwargs.get("video_path")
        size = create_kwargs.get("size", 512)

        if video_path is None:
            raise ValueError("Missing required 'video_path' parameter in create_kwargs")

        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        # Load video to get metadata
        try:
            vr = torchcodec.decoders.VideoDecoder(video_path)
            duration = vr.metadata.duration_seconds
            duration = np.floor(duration)
        except Exception as e:
            raise ValueError(f"Failed to load video from {video_path}: {e}") from e

        self._instance_dict[instance_id] = {
            "size": size,
            "duration": duration,
            "decoder": vr,
        }

        logger.info(f"Created ZoomInFrameTool instance {instance_id} for video: {video_path} (duration: {duration}s)")

        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute frame extraction and zoom at a specific timestamp.

        Args:
            instance_id: The instance id of the tool
            parameters: Dict containing 'timestamp' string in HH:MM:SS format
            **kwargs: Additional execution kwargs

        Returns:
            Tuple of (ToolResponse with extracted frame, reward, metrics)
        """
        timestamp_str = parameters.get("timestamp")

        if not timestamp_str:
            error_msg = "Error: 'timestamp' parameter is required in HH:MM:SS format."
            logger.warning(f"Tool execution failed: {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        instance_data = self._instance_dict[instance_id]
        size = instance_data["size"]
        duration = instance_data["duration"]
        duration_str = encode_timestamp_to_str(duration)
        vr = instance_data["decoder"]

        try:
            # Parse timestamp
            timestamp_sec = decode_timestamp_to_str(timestamp_str)
            max_sec = vr.metadata.end_stream_seconds - 1e-6
            min_sec = vr.metadata.begin_stream_seconds
            timestamp_sec = np.clip(timestamp_sec, min_sec, max_sec).item()

            # Validate timestamp
            if timestamp_sec < 0 or timestamp_sec > duration:
                error_msg = (
                    f"Error: Timestamp out of range. Video duration is {duration_str}, "
                    f"but requested timestamp={timestamp_str}"
                )
                logger.warning(error_msg)
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            # Extract frame at this timestamp
            try:
                frames = vr.get_frames_played_at([timestamp_sec]).data  # Returns torch tensor: 1 x C x H x W
            except Exception as e:
                print(f"Error extracting frame: {timestamp_sec} {min_sec} {max_sec}")
                raise e

            logger.info(f"Extracted frame at {timestamp_str}, shape: {frames.shape}, dtype: {frames.dtype}")

            # Convert to PIL Image with resizing
            pil_images = torch_frames_to_pil_images(frames, size=size)

            response_text = f"Zoomed in on frame, Frame with timestamp: {timestamp_str} <image>."

            return (
                ToolResponse(image=pil_images, text=response_text),
                0.0,
                {"success": True, "timestamp": timestamp_str},
            )

        except Exception as e:
            error_msg = f"Error extracting frame: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResponse(text=error_msg), -0.05, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance and cleanup resources.

        Args:
            instance_id: The instance id of the tool
        """
        if instance_id in self._instance_dict:
            # Clean up decoder if needed
            instance_data = self._instance_dict[instance_id]
            if "decoder" in instance_data:
                del instance_data["decoder"]
            del self._instance_dict[instance_id]
            logger.info(f"Released ZoomInFrameTool instance {instance_id}")


class VerifyAnswerTool(BaseTool):
    """A tool for verifying the final answer against ground truth.

    This tool compares the model's answer with the ground truth answer
    and calculates a reward based on correctness using the video_holmes scorer.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance with ground_truth
        execute: Execute answer verification
        calc_reward: Calculate the reward using video_holmes scorer
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize VerifyAnswerTool.

        Tool schema example:
        {
            "type": "function",
            "function": {
                "name": "verify_answer",
                "description": "A tool for verifying the final answer against the ground truth",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "The answer to verify"}
                    },
                    "required": ["answer"]
                }
            }
        }

        Args:
            config: Tool configuration dict
            tool_schema: OpenAI function tool schema
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        logger.info(f"Initialized VerifyAnswerTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance for answer verification.

        Args:
            instance_id: Optional unique identifier for the instance
            **kwargs: Should contain 'create_kwargs' dict with:
                - ground_truth: The ground truth answer (required, typically an option letter like 'A', 'B', etc.)

        Returns:
            Tuple of (instance_id, ToolResponse)

        Raises:
            ValueError: If ground_truth is missing
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Extract create_kwargs from kwargs
        create_kwargs = kwargs.get("create_kwargs", {})
        
        # Extract ground_truth from create_kwargs
        ground_truth = create_kwargs.get("ground_truth")

        if ground_truth is None:
            raise ValueError("Missing required 'ground_truth' parameter in create_kwargs")

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }

        logger.info(f"Created VerifyAnswerTool instance {instance_id} with ground_truth: {ground_truth}")

        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute answer verification.

        Args:
            instance_id: The instance id of the tool
            parameters: Dict containing 'answer' string to verify
            **kwargs: Additional execution kwargs

        Returns:
            Tuple of (ToolResponse with verification result, tool reward, metrics)
        """
        answer = parameters.get("answer", "")

        if not isinstance(answer, str):
            answer = str(answer)

        # Store the response
        self._instance_dict[instance_id]["response"] = answer

        # Calculate reward using video_holmes scorer
        reward = await self.calc_reward(instance_id)

        # Penalty for non-improved answer submission (same pattern as gsm8k_tool)
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05

        # Update the reward
        self._instance_dict[instance_id]["reward"] = reward

        logger.info(f"Verified answer '{answer}' with reward={reward}")

        return (
            ToolResponse(text=f"Current parsed answer={answer}, reward={reward}"),
            tool_reward,
            {"success": True, "reward": reward},
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward using video_holmes scorer.

        Args:
            instance_id: The instance id of the tool
            **kwargs: Additional kwargs

        Returns:
            Reward score (float)
        """
        return video_holmes.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            logger.info(f"Released VerifyAnswerTool instance {instance_id}")
