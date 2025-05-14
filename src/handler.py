"""Example handler file."""

import os
import runpod
import io
import audiofile
from uvr import models
import torchaudio
import base64
import torch

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

demucs = models.Demucs(
    name="hdemucs_mmi", other_metadata={"segment": 2, "split": True}, device=device
)


def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]

    audio_base64 = job_input.get("audio")
    if audio_base64 is None:
        raise ValueError("Missing 'audio' key in input.")

    split_parts = job_input.get("parts", ["vocals", "bass", "drums", "other"])
    print(f"Splitting into parts: {split_parts}")

    tmp_audio_file_name = "/tmp/" + str(hash(audio_base64)) + ".mp3"
    with open(tmp_audio_file_name, "wb") as audio_file:
        buffer_ = io.BytesIO(bytes(audio_base64, "utf-8"))
        base64.decode(buffer_, audio_file)

    try:
        res = demucs(tmp_audio_file_name)
        _, rate = audiofile.read(tmp_audio_file_name)
    finally:
        os.remove(tmp_audio_file_name)

    result = {}
    for part in split_parts:
        buffer_ = io.BytesIO()
        torchaudio.save(
            buffer_,
            res[part],
            rate,
            format="ogg",
        )
        buffer_.seek(0)
        result[part] = base64.b64encode(buffer_.read()).decode("utf-8")

    return result


runpod.serverless.start({"handler": handler})
