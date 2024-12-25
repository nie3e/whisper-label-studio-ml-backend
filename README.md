# ASR with Whisper for Label Studio ML Backend

This project uses Whisper to transcribe data from Label Studio project.

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart). 

## Labeling interface

This project works with the Label Studio's pre-built `Automatic Speech Recognition using Segments` template (available under `Audio/Speech Processing > Automatic Speech Recognition using Segments`)

```xml
<View>
  <Labels name="labels" toName="audio">
    <Label value="Speech" />
    <Label value="Noise" />
  </Labels>
  <Audio name="audio" value="$audio"/>
  <TextArea name="transcription" toName="audio" rows="2" editable="true" perRegion="true" required="true" />
</View>
```

> Warning: If you use files hosted in Label Studio (meaning they were added using the import action), hosted in cloud storage, or connected through local storage, then you must provide the `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY` environment variables to the ML backend. For more information, see [Allow the ML backend to access Label Studio data](https://labelstud.io/guide/ml#Allow-the-ML-backend-to-access-Label-Studio-data). For information about finding your Label Studio API key, see [Access token](https://labelstud.io/guide/user_account#Access-token).

## Running with Docker (recommended)

1. Start the Machine Learning backend on `http://localhost:9090` with the prebuilt image:

```bash
docker-compose up
```

2. Validate that backend is running:

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Create a project in Label Studio. Then from the **Model** page in the project settings, [connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio). The default URL is `http://localhost:9090`.

## Configuration

Parameters can be set in `docker-compose.yml` before running the container.

The following common parameters are available:
- `MODEL_NAME` - Specify the model name for the Whisper.
- `BASIC_AUTH_USER` - Specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - Specify the basic auth password for the model server
- `LOG_LEVEL` - Set the log level for the model server
- `WORKERS` - for now it should be 1
- `THREADS` - for now it should be 1
- `LABEL_STUDIO_HOST`: The host of the Label Studio instance. Default is `http://localhost:8080`. Change to `http://host.docker.internal:8080` if you run Label Studio from docker.
- `LABEL_STUDIO_API_KEY`: The API key for the Label Studio instance.
- `BATCH_SIZE` - Batch size for pipeline. Default is 8.

## Known issues and TODOs
- CUDA problems when using workers/threads greater than 1
- Inference stucks when using clean CMD command from `Dockerfile`
- Adding the ability to select a language (for now it is whisper-auto based on first 30 seconds)

## References

I relied on the following examples from [label-studio-ml-backend](https://github.com/HumanSignal/label-studio-ml-backend):

- [examples/huggingface_llm](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/huggingface_llm) - How to use transformers with ML Backend
- [examples/nemo_asr](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/nemo_asr) - How to use ASR with ML Backend