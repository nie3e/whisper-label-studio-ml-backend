"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""

import pytest
import json
from model import Whisper


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=Whisper)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = json.loads(
        '{"tasks": [{"id": 102651352, "data": {"audio": '
        '"https://htx-pub.s3.amazonaws.com/datasets/audio/f2bjrop1.1.wav"}, '
        '"meta": {}, "created_at": "2024-04-08T15:52:07.473205Z", '
        '"updated_at": "2024-04-08T15:52:07.473210Z", "is_labeled": false, '
        '"overlap": 1, "inner_id": 2, "total_annotations": 0, '
        '"cancelled_annotations": 0, "total_predictions": 0, '
        '"comment_count": 0, "unresolved_comment_count": 0, '
        '"last_comment_updated_at": null, "project": 62075, "updated_by": '
        'null, "file_upload": 1430391, "comment_authors": [], "annotations": '
        '[], "predictions": []}], "project": "62075.1712591487", '
        '"label_config": "<View>\\n  <Audio name=\\"audio\\" '
        'value=\\"$audio\\" zoom=\\"true\\" hotkey=\\"ctrl+enter\\" />\\n  '
        '<Header value=\\"Provide Transcription\\" />\\n  <TextArea '
        'name=\\"transcription\\" toName=\\"audio\\"\\n            '
        'rows=\\"4\\" editable=\\"true\\" maxSubmissions=\\"1\\" '
        '/>\\n</View>", "params": {"login": null, "password": null, '
        '"context": null}}'
    )

    expected_response = json.loads(
        '{"results": [{"model_version": "Whisper-v0.0.1", "result": [{'
        '"from_name": "labels", "id": "0", "to_name": "audio", "type": '
        '"labels", "value": {"channel": 0, "end": 2.0, "labels": ["Speech"], '
        '"start": 0.0}}, {"from_name": "transcription", "id": "0", '
        '"to_name": "audio", "type": "textarea", "value": {"channel": 0, '
        '"end": 2.0, "start": 0.0, "text": [" i w tym momencie, \u017ce jest '
        'to, co nie jest w tym momencie,"]}}, {"from_name": "labels", '
        '"id": "1", "to_name": "audio", "type": "labels", "value": {'
        '"channel": 0, "end": 4.0, "labels": ["Speech"], "start": 2.0}}, '
        '{"from_name": "transcription", "id": "1", "to_name": "audio", '
        '"type": "textarea", "value": {"channel": 0, "end": 4.0, "start": '
        '2.0, "text": [" to jest to, co nie jest w tym momencie,"]}}, '
        '{"from_name": "labels", "id": "2", "to_name": "audio", "type": '
        '"labels", "value": {"channel": 0, "end": 6.0, "labels": ["Speech"], '
        '"start": 4.0}}, {"from_name": "transcription", "id": "2", '
        '"to_name": "audio", "type": "textarea", "value": {"channel": 0, '
        '"end": 6.0, "start": 4.0, "text": [" co nie jest w tym momencie,'
        '"]}}, {"from_name": "labels", "id": "3", "to_name": "audio", '
        '"type": "labels", "value": {"channel": 0, "end": 8.0, "labels": ['
        '"Speech"], "start": 6.0}}, {"from_name": "transcription", '
        '"id": "3", "to_name": "audio", "type": "textarea", "value": {'
        '"channel": 0, "end": 8.0, "start": 6.0, "text": [" co nie jest w '
        'tym momencie,"]}}, {"from_name": "labels", "id": "4", "to_name": '
        '"audio", "type": "labels", "value": {"channel": 0, "end": 10.0, '
        '"labels": ["Speech"], "start": 8.0}}, {"from_name": '
        '"transcription", "id": "4", "to_name": "audio", "type": "textarea", '
        '"value": {"channel": 0, "end": 10.0, "start": 8.0, "text": [" co '
        'nie jest w tym momencie,"]}}, {"from_name": "labels", "id": "5", '
        '"to_name": "audio", "type": "labels", "value": {"channel": 0, '
        '"end": 12.0, "labels": ["Speech"], "start": 10.0}}, {"from_name": '
        '"transcription", "id": "5", "to_name": "audio", "type": "textarea", '
        '"value": {"channel": 0, "end": 12.0, "start": 10.0, "text": [" co '
        'nie jest w tym momencie,"]}}, {"from_name": "labels", "id": "6", '
        '"to_name": "audio", "type": "labels", "value": {"channel": 0, '
        '"end": 14.0, "labels": ["Speech"], "start": 12.0}}, {"from_name": '
        '"transcription", "id": "6", "to_name": "audio", "type": "textarea", '
        '"value": {"channel": 0, "end": 14.0, "start": 12.0, "text": [" co '
        'nie jest w tym momencie,"]}}, {"from_name": "labels", "id": "7", '
        '"to_name": "audio", "type": "labels", "value": {"channel": 0, '
        '"end": 16.0, "labels": ["Speech"], "start": 14.0}}, {"from_name": '
        '"transcription", "id": "7", "to_name": "audio", "type": "textarea", '
        '"value": {"channel": 0, "end": 16.0, "start": 14.0, "text": [" co '
        'nie jest w tym momencie,"]}}, {"from_name": "labels", "id": "8", '
        '"to_name": "audio", "type": "labels", "value": {"channel": 0, '
        '"end": 18.0, "labels": ["Speech"], "start": 16.0}}, {"from_name": '
        '"transcription", "id": "8", "to_name": "audio", "type": "textarea", '
        '"value": {"channel": 0, "end": 18.0, "start": 16.0, "text": [" co '
        'nie jest w tym momencie,"]}}, {"from_name": "labels", "id": "9", '
        '"to_name": "audio", "type": "labels", "value": {"channel": 0, '
        '"end": 20.0, "labels": ["Speech"], "start": 18.0}}, {"from_name": '
        '"transcription", "id": "9", "to_name": "audio", "type": "textarea", '
        '"value": {"channel": 0, "end": 20.0, "start": 18.0, "text": [" co '
        'nie jest w tym momencie,"]}}, {"from_name": "labels", "id": "10", '
        '"to_name": "audio", "type": "labels", "value": {"channel": 0, '
        '"end": 22.0, "labels": ["Speech"], "start": 20.0}}, {"from_name": '
        '"transcription", "id": "10", "to_name": "audio", "type": '
        '"textarea", "value": {"channel": 0, "end": 22.0, "start": 20.0, '
        '"text": [" co nie jest w tym momencie,"]}}, {"from_name": "labels", '
        '"id": "11", "to_name": "audio", "type": "labels", "value": {'
        '"channel": 0, "end": 24.0, "labels": ["Speech"], "start": 22.0}}, '
        '{"from_name": "transcription", "id": "11", "to_name": "audio", '
        '"type": "textarea", "value": {"channel": 0, "end": 24.0, "start": '
        '22.0, "text": [" co nie jest w tym momencie,"]}}, {"from_name": '
        '"labels", "id": "12", "to_name": "audio", "type": "labels", '
        '"value": {"channel": 0, "end": 26.0, "labels": ["Speech"], "start": '
        '24.0}}, {"from_name": "transcription", "id": "12", "to_name": '
        '"audio", "type": "textarea", "value": {"channel": 0, "end": 26.0, '
        '"start": 24.0, "text": [" co nie jest w tym momencie,"]}}, '
        '{"from_name": "labels", "id": "13", "to_name": "audio", "type": '
        '"labels", "value": {"channel": 0, "end": 28.0, "labels": ['
        '"Speech"], "start": 26.0}}, {"from_name": "transcription", '
        '"id": "13", "to_name": "audio", "type": "textarea", "value": {'
        '"channel": 0, "end": 28.0, "start": 26.0, "text": [" co nie jest w '
        'tym momencie,"]}}], "score": 0.0}]}'
    )

    response = client.post('/predict', data=json.dumps(request),
                           content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    assert response == expected_response
