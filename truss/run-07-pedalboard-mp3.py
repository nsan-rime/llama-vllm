import asyncio

from model.model import Model

if __name__ == "__main__":

    m = Model()
    m.load()

    output = m.predict({"text":"this is a test", "audioFormat": "mp3"})
    
    output = asyncio.run(output)

    async def collect_chunks():
        chunks = []
        async for chunk in output.body_iterator:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect_chunks())

    with open("/workspace/tmp/test-pedalboard-audio.mp3", "wb") as f:
        for chunk in chunks:
            f.write(chunk)

    print("=== Saved /workspace/tmp/test-pedalboard-audio.mp3 ===")
