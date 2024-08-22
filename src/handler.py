""" Example handler file. """

import os
import runpod
import io
import audiofile
from uvr import models
import torchaudio
import base64

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

device = os.getenv("DEVICE", "cpu")
demucs = models.Demucs(
    name="hdemucs_mmi",
    other_metadata={"segment": 2, "split": True},
    device=device)

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    audio_base64 = job_input.get('audio')
    if audio_base64 is None:
        raise ValueError("Missing 'audio' key in input.")

    tmp_audio_file_name = '/tmp/' + str(hash(audio_base64)) + '.mp3'
    with open(tmp_audio_file_name, 'wb') as audio_file:
        buffer_ = io.BytesIO(bytes(audio_base64, 'utf-8'))
        base64.decode(buffer_, audio_file)

    res = demucs(tmp_audio_file_name)
    _, rate = audiofile.read(tmp_audio_file_name)

    result = {}
    split_parts = ['vocals', 'bass', 'drums', 'other']
    tmp_audio_part_name_template = '/tmp/' + str(hash(audio_base64)) + '{}.mp3'
    for part in split_parts:
        torchaudio.save(
            tmp_audio_part_name_template.format(part), res[part],
            rate
        )
        with open(tmp_audio_part_name_template.format(part), 'rb') as pfile:
            result[part] = base64.b64encode(pfile.read()).decode('utf-8')

    return result


runpod.serverless.start({"handler": handler})
