import logging
import os
import subprocess
import asyncio
import torch
import struct
from fastapi.responses import StreamingResponse
import time
from fastapi import HTTPException

import numpy as np
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from typing import Optional
from transformers import AutoTokenizer, AutoConfig
from snac import SNAC
from safetensors.torch import load_file
from pedalboard.io import AudioFile

from uuid_extensions import uuid7str

import json
import yaml

# mp3
from pydub import AudioSegment
from io import BytesIO # ffmpeg-python

def warmup_snac(model, device="cuda"):
    num_frames = 8  # 8 frames × 7 tokens = 56 tokens
    codes_0 = torch.zeros(1, num_frames, dtype=torch.int32, device=device)
    codes_1 = torch.zeros(1, 2 * num_frames, dtype=torch.int32, device=device)
    codes_2 = torch.zeros(1, 4 * num_frames, dtype=torch.int32, device=device)

    dummy_codes = [codes_0, codes_1, codes_2]
    _ = model.decode(dummy_codes)  # This triggers torch.compile trace

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda"
model = model.to(snac_device)
model.decode = torch.compile(model.decode)
start = time.time()
warmup_snac(model)
print(f"Warming up SNAC took {time.time() - start}")

logger = logging.getLogger(__name__)


def convert_to_audio(multiframe, count):
    """Convert a list of token IDs into audio bytes efficiently."""
    if len(multiframe) < 7:
        print("Returning None, didn't generate audio due to multiframe length less than seven")
        return None
    
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

    codes_0 = torch.zeros(num_frames, device=snac_device, dtype=torch.int32)
    codes_1 = torch.zeros(2 * num_frames, device=snac_device, dtype=torch.int32)
    codes_2 = torch.zeros(4 * num_frames, device=snac_device, dtype=torch.int32)

    for j in range(num_frames):
        i = 7 * j
        codes_0[j] = frame[i]
        codes_1[2 * j] = frame[i + 1]
        codes_1[2 * j + 1] = frame[i + 4]
        codes_2[4 * j] = frame[i + 2]
        codes_2[4 * j + 1] = frame[i + 3]
        codes_2[4 * j + 2] = frame[i + 5]
        codes_2[4 * j + 3] = frame[i + 6]

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        print("Returning None, didn't generate audio due to codes out of range.")
        return None

    with torch.inference_mode():
        audio_hat = model.decode(codes)
    
    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return audio_bytes, audio_np[0,0]

def turn_token_into_id(token_string, index):
    """Extract and convert the last custom token ID from a string."""
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        print("No token found in the string")
        return None
    
    last_token = token_string[last_token_start:]
    token_to_remove = "<custom_token_2>"
    if token_to_remove in last_token:
        print("Token to remove found in last token")
        return None
    if last_token.startswith("<custom_token_") and last_token.endswith(">") and token_to_remove not in last_token:
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None

async def tokens_decoder(token_gen):
    """Async generator to decode tokens into audio chunks."""
    buffer = []
    count = 0
    yielded_first_token = False
    async for token_sim in token_gen:
        start = time.time()
        token = turn_token_into_id(token_sim, count)
        if token is not None and token >= 0:
            buffer.append(token)
            count += 1
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                print(f"Time to generate audio: {time.time() - start}")
                if audio_samples is not None:
                    yield audio_samples

    # after the stream ends, yield any remaining tokens if buffer has leftovers
    if count > 27:
        remaining = buffer[-28:]
    else:
        remaining = buffer

    if remaining:
        audio_samples = convert_to_audio(remaining, count)
        if audio_samples is not None:
            yield audio_samples

class BytesIOBuffer:
    """
    A wrapper around a BytesIO that allows a writer to write
    and a reader to read, treating the BytesIO as a queue.
    """

    def __init__(self):
        self.write_buf = BytesIO()
        self.pointer = 0

    def read(self, n=None):
        old_pos = self.write_buf.tell()
        self.write_buf.seek(self.pointer)
        chunk = self.write_buf.read(n)
        new_pos = self.write_buf.tell()
        self.write_buf.seek(old_pos)
        self.pointer = new_pos
        return chunk

class OrpheusModel:
    def __init__(self, 
                 model_path,
                 dtype=torch.bfloat16,
                 seed: int = 0,
                 max_model_len: Optional[int] = 8_192, # Nay changed this
                 cpu_offload_gb: float = 0, ### TBD 4/18 during load testing
                 gpu_memory_utilization: float = 0.5, ### Nay
                 quantization: Optional[str] = None,
                 max_seq_len_to_capture: int = 8192,
                 enforce_eager: Optional[bool] = None):
        self.model_path = model_path  # Store local path
        self.dtype = dtype
        self.engine = self._setup_engine(seed, max_model_len, cpu_offload_gb, 
                                        gpu_memory_utilization, quantization, 
                                        max_seq_len_to_capture, enforce_eager)

        self.tokeniser = asyncio.run(self.engine.get_tokenizer())

    # def _load_local_tokenizer(self):
    #     """Load tokenizer from local files"""
    #     return AutoTokenizer.from_pretrained(
    #         self.model_path,
    #         local_files_only=True
    #     )
    
    def _setup_engine(self, seed, max_model_len, cpu_offload_gb, gpu_memory_utilization, 
                      quantization, max_seq_len_to_capture, enforce_eager):
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            dtype=self.dtype,
            max_model_len=max_model_len,
            cpu_offload_gb=cpu_offload_gb,
            gpu_memory_utilization=gpu_memory_utilization, ## LOOK AT THIS!
            quantization=quantization,
            max_seq_len_to_capture=max_seq_len_to_capture,
            enforce_eager=enforce_eager,
            seed=seed
        )

        return AsyncLLMEngine.from_engine_args(engine_args)

    def _format_prompt(self, prompt, speaker=None):
        if speaker:
            print("***************************YES speaker")
            adapted_prompt = "{" + f"{speaker}" + "}: " + f"{prompt}"
            prompt_tokens = self.tokeniser(adapted_prompt, return_tensors="pt")
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
            all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
            return self.tokeniser.decode(all_input_ids[0])
        else:
            prompt_tokens = self.tokeniser(prompt, return_tensors="pt")
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            # end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
            all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
            return self.tokeniser.decode(all_input_ids[0])

    async def generate_tokens(self, prompt, speaker=None, request_id=None, 
                              temperature=0.6, top_p=0.8, max_tokens=1200, 
                              stop_token_ids=[128258], repetition_penalty=1.3):
        print(f"User Prompt === {prompt}")
        print(f"Model Parameters === speaker: {speaker}, temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, repetition_penalty: {repetition_penalty}, stop_token_ids: {stop_token_ids} ")
        prompt_string = self._format_prompt(prompt, speaker)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )
        time_before_first_generated_token = time.time()
        generated_token_count = 0
        async for result in self.engine.generate(prompt=prompt_string, 
                                                 sampling_params=sampling_params, 
                                                 request_id=request_id):
            if generated_token_count == 0:
                time_after_last_token = time.time()
                print(f"Generating token... Time to generate token number {generated_token_count}: {time_after_last_token - time_before_first_generated_token}")
                generated_token_count += 1
            else:
                print(f"Generating token... Time to generate token number {generated_token_count}: {time.time() - time_after_last_token}")
                time_after_last_token = time.time()
                generated_token_count += 1
            yield result.outputs[0].text

class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.orpheus_model_path = None
        self._secrets = kwargs.get("secrets")
        
        # self._environment = kwargs.get("environment", {})
        # self.env_name = self._environment.get("name") if self._environment else None
        # self.data_dir=kwargs["data_dir"]
        # self.model_metadata = kwargs.get("config", {}).get("model_metadata", {})

        # deploy_cfg_path = root / self.model_metadata.get("deploy_cfg")        
        # assert deploy_cfg_path.exists(), ValueError(f"Deployment config file not found at {deploy_cfg_path}")

        # self._s3 = S3Helper(self.data_dir, self.model_metadata)
        
        # with open(deploy_cfg_path) as f:
        #     self.deploy_cfg = yaml.safe_load(f)
        # for model_name, model_cfg in self.deploy_cfg["models"].items():
        #     if model_name == "snac":
        #         self.snac_config_path = model_cfg["config_path"]
        #         self.snac_model_path = model_cfg["checkpoint_path"]
        #     elif model_name == "orpheus":
        #         self.orpheus_model_path = model_cfg["model_dir"]

    def load(self):
        if self._secrets:
            # vLLM expects HF_TOKEN to be configured via an environment variable
            os.environ["HF_TOKEN"] = self._secrets["hf_orpheus_access_token_nay"]
            
        # self._s3.download_files()
        self.model = OrpheusModel(model_path="canopylabs/orpheus-3b-0.1-ft", 
                                  dtype=torch.float16)
    
    def create_wav_header(self, sample_rate=24000, bits_per_sample=16, channels=1):
                """Creates a WAV file header."""
                byte_rate = sample_rate * channels * bits_per_sample // 8
                block_align = channels * bits_per_sample // 8
                data_size = 0
                header = struct.pack(
                    '<4sI4s4sIHHIIHH4sI',
                    b'RIFF', 36 + data_size, b'WAVE', b'fmt ', 16, 1, channels,
                    sample_rate, byte_rate, block_align, bits_per_sample, b'data', data_size
                )
                return header

    async def predict(self, model_input):
        print("Starting predict call")
        start_predict = time.time()
        # async def start_ffmpeg_encoder():
        #     cmd = [
        #         "ffmpeg",
        #         "-f", "s16le",
        #         "-ar", "24000",
        #         "-ac", "1",
        #         "-i", "pipe:0",
        #         "-f", "mp3",
        #         "-flush_packets", "1",
        #         "-b:a", "128k",
        #         "pipe:1"
        #     ]


        #     proc = await asyncio.create_subprocess_exec(
        #         *cmd,   
        #         stdin=asyncio.subprocess.PIPE,
        #         stdout=asyncio.subprocess.PIPE,
        #         stderr=asyncio.subprocess.DEVNULL
        #     )
        #     return proc

        respondStreaming = model_input.get("respondStreaming", True)
        if not respondStreaming:
            raise HTTPException(status_code=400, detail="Streaming mode is required.")

        """Async predict method to stream audio for concurrent requests."""
        text = str(model_input.get("text", "Hi, I'm Orpheus model"))
        speaker = model_input.get("speaker", None)
        request_id = str(model_input.get("request_id", uuid7str()))
        repetition_penalty = model_input.get("repetition_penalty", 1.25)
        max_tokens = int(model_input.get("max_tokens", 5000))
        temperature = model_input.get("temperature", 0.1)
        top_p = model_input.get("top_p", .75)
        audioFormat = model_input.get("audioFormat", "wav")

        logger.info(f"Generating audio from processed text ({len(text)} chars, speaker {speaker}): {text}")
        if audioFormat == "mp3":
            async def audio_stream():
                start_audio_stream = time.time()
                print(f"Starting audio stream after {start_audio_stream - start_predict}")

                token_gen = self.model.generate_tokens(
                    prompt=text,
                    speaker=speaker,
                    request_id=request_id,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_token_ids=[128258],
                )

                buffer = BytesIOBuffer()

                with AudioFile(buffer.write_buf, mode="w", format="mp3", samplerate=24_000, num_channels=1, bit_depth=16, quality="fastest") as f:
                    # Yield mp3 header
                    yield b'ID3\x04\x00\x00\x00\x00\x00#TSSE\x00\x00\x00\x0f\x00\x00\x03Lavf60.16.100'

                    async for _, audio_np in tokens_decoder(token_gen):
                        f.write(audio_np)
                        encoded_bytes = buffer.read()

                        if len(encoded_bytes) > 0:
                            yield encoded_bytes
                        
            return StreamingResponse(
                audio_stream(), 
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline; filename=audio.mp3"} # optional but good for browsers/clients
            )
        elif audioFormat == "wav":

            async def audio_stream():
                yield self.create_wav_header()

                token_gen = self.model.generate_tokens(
                    prompt=text,
                    speaker=speaker,
                    request_id=request_id,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_token_ids=[128258],
                )

                buffer_chunks = []
                async for audio_chunk, _ in tokens_decoder(token_gen):
                    buffer_chunks.append(audio_chunk)

                    # Once we have 3 buffered, start yielding them and then stream live
                    if len(buffer_chunks) == 1:
                        for chunk in buffer_chunks:
                            yield chunk
                        buffer_chunks.clear()
                    elif len(buffer_chunks) > 1:
                        yield buffer_chunks.pop(0)  # keep buffer rolling by 1

                # Yield any remaining chunks in the buffer at the end
                for chunk in buffer_chunks:
                    yield chunk

            return StreamingResponse(
                audio_stream(),
                media_type="audio/wav"
            )
