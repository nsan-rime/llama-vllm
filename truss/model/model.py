import asyncio
import json
import os
import torch
import uuid

from fastapi.responses import StreamingResponse

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Find .project-root file to set location of root directory for imports
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False, indicator=[".project-root"])

from snac_decode import create_wav_header, tokens_decoder

class Model:
    def __init__(self, **kwargs) -> None:
        self._secrets = kwargs.get("secrets")

        self.engine_args = {
            "model" : "canopylabs/orpheus-3b-0.1-ft"
        }

    def load(self) -> None:
        if self._secrets:
            # vLLM expects HF_TOKEN to be configured via an environment variable
            os.environ["HF_TOKEN"] = self._secrets["hf_orpheus_access_token_nay"]

        self.llm_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(**self.engine_args)
        )
        self.tokenizer = asyncio.run(self.llm_engine.get_tokenizer())

    def build_prompt(self, text):
        prompts = [text]
        all_input_ids = []
        
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            all_input_ids.append(input_ids)

        start_token = torch.tensor([[ 128_259 ]], dtype=torch.int64)          # Start of human
        end_tokens = torch.tensor([[ 128_009, 128_260 ]], dtype=torch.int64)  # End of text, End of human

        modified_input_ids = torch.cat([start_token, all_input_ids[0], end_tokens], dim=1)
        prompt_string = self.tokenizer.decode(modified_input_ids[0])
        return prompt_string, modified_input_ids[0].size(0)

    async def generate_tokens(self, request):

        prompt_string, prompt_ntokens = self.build_prompt(request["text"])

        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=1_200,
            stop_token_ids=[ 128_258 ],
            repetition_penalty=1.3
        )

        idx = str(uuid.uuid4().hex)

        vllm_generator = self.llm_engine.generate(
            prompt_string,
            sampling_params,
            idx
        )

        async for result in vllm_generator:
            yield result.outputs[0].text

    async def predict(self, request):

        async def audio_stream():
            yield create_wav_header()

            token_gen = self.generate_tokens(request)

            buffer_chunks = []
            async for audio_chunk in tokens_decoder(token_gen):
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
