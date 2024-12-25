import torch
import os
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import \
    get_local_path
from transformers import pipeline
import torchaudio


MODEL_NAME = os.getenv("MODEL_NAME", "openai/whisper-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
_model = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
    torch_dtype=torch.bfloat16,
)


def _get_task_result(chunks: list[dict]) -> list[dict]:
    """Changes whisper result into Label Studio predictions

    Args:
        chunks (list[dict]): list of transcription segments

    Returns:
        (list[dict]): Label Studio predictions-like object
    """
    task_result = []
    for i, chunk in enumerate(chunks):
        start_t, end_t = chunk["timestamp"]
        text = chunk["text"]
        result_label = {
            "id": str(i),
            "type": "labels",
            "value": {
                "end": end_t, "start": start_t, "labels": ["Speech"],
                "channel": 0
            },
            "from_name": "labels",
            "to_name": "audio"
        }
        result_segment = {
            "id": str(i),
            "type": "textarea",
            "value": {
                "end": end_t, "start": start_t, "text": [text],
                "channel": 0
            },
            "from_name": "transcription",
            "to_name": "audio"
        }
        task_result.append(result_label)
        task_result.append(result_segment)
    return task_result


class Whisper(LabelStudioMLBase):
    """Custom ML Backend model
    """
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))

    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None,
                **kwargs) -> ModelResponse:
        from_name, to_name, value = (
            self.label_interface.get_first_tag_occurence(
                'TextArea', 'Audio'
            )
        )
        audio_paths = []
        for task in tasks:
            audio_url = task['data'].get(value) or task['data'].get(
                DATA_UNDEFINED_NAME)
            audio_path = get_local_path(audio_url, task_id=task.get('id'))
            audio_paths.append(audio_path)

        predictions = []
        for audio_path in audio_paths:
            sample, sr = torchaudio.load(audio_path)
            sample = torchaudio.functional.resample(
                sample, orig_freq=sr, new_freq=16000
            ).numpy()

            result = _model(
                sample[0].copy(),
                batch_size=self.BATCH_SIZE,
                return_timestamps=True,
                generate_kwargs={
                    "task": "transcribe",
                    "do_sample": False
                }
            )

            task_result = _get_task_result(result["chunks"])
            predictions.append({"result": task_result})

        return ModelResponse(
            predictions=predictions,  # noqa
            model_version=self.get("model_version")
        )
