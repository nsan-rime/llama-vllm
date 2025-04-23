import struct
import time
import torch

import numpy as np

from snac import SNAC

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
    return audio_bytes

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

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
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
