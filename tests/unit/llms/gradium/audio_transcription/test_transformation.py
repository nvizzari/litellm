import io
import json
import wave
from typing import Final
from urllib.parse import parse_qs

import httpx
import pytest
import respx

import litellm
from litellm.llms.gradium.common_utils import GradiumException
from litellm.types.utils import TranscriptionResponse

_API_BASE: Final = "https://eu.api.gradium.ai"
_API_KEY: Final = "gd_test_key"
_URL: Final = f"{_API_BASE}/api/speech/asr"
_SAMPLE_RATE: Final = 24_000


def _wav(seconds: float) -> bytes:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(_SAMPLE_RATE)
        audio.writeframes(b"\x00\x00" * int(seconds * _SAMPLE_RATE))
    return buffer.getvalue()


def _ndjson(*messages: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        content="\n".join(json.dumps(message) for message in messages).encode(),
        headers={"content-type": "application/x-ndjson"},
    )


_HELLO_WORLD_MESSAGES: Final = (
    {"type": "text", "text": "Hello", "start_s": 0.5, "stream_id": 0},
    {"type": "end_text", "stop_s": 1.0, "stream_id": 0},
    {"type": "text", "text": " world.", "start_s": 1.2, "stream_id": 0},
    {"type": "end_text", "stop_s": 2.5, "stream_id": 0},
)
_HELLO_WORLD: Final = _ndjson(*_HELLO_WORLD_MESSAGES)


def _processed(*total_durations_s: float) -> httpx.Response:
    steps: Final[tuple[dict[str, object], ...]] = tuple(
        {"type": "step", "vad": [], "step_idx": 0, "step_duration_s": 0.08, "total_duration_s": total}
        for total in total_durations_s
    )
    return _ndjson(*steps, *_HELLO_WORLD_MESSAGES)


@pytest.fixture
def gradium_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def _transcribe(file: tuple[str, bytes] = ("speech.wav", b"RIFFaudio"), **params: object) -> TranscriptionResponse:
    response: Final = litellm.transcription(
        model="gradium/default", file=file, api_base=_API_BASE, api_key=_API_KEY, **params
    )
    assert isinstance(response, TranscriptionResponse)
    return response


def _json_config(route: respx.Route) -> dict[str, object]:
    query: Final = parse_qs(route.calls[-1].request.url.query.decode())
    return json.loads(query["json_config"][0])


# Gradium's REST ASR takes the raw audio as the body, labelled by Content-Type, and its
# settings as query parameters, not the multipart form OpenAI callers send.
def test_sends_the_audio_as_a_raw_body_with_settings_in_the_query(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(url__startswith=_URL).mock(return_value=_HELLO_WORLD)

        _transcribe(file=("speech.wav", b"RIFFaudio"), language="fr", temperature=0.3)

        request: Final = route.calls[-1].request
        assert request.content == b"RIFFaudio"
        assert request.headers["content-type"] == "audio/wav"
        assert request.headers["x-api-key"] == _API_KEY
        assert parse_qs(request.url.query.decode())["model"] == ["default"]
        assert _json_config(route) == {"language": "fr", "temp": 0.3}


# Gradium rejects requests without a language, while OpenAI callers may leave it out to auto-detect.
def test_asks_gradium_to_detect_the_language_when_the_caller_sets_none(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(url__startswith=_URL).mock(return_value=_HELLO_WORLD)

        _transcribe()

        assert _json_config(route) == {"language": "any"}


# The proxy hands multipart form fields over as strings, and Gradium silently ignores a setting
# whose JSON type is wrong (delay_in_frames "16" falls back to its default), so they must be decoded.
def test_decodes_settings_that_arrive_as_form_strings(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(url__startswith=_URL).mock(return_value=_HELLO_WORLD)

        _transcribe(
            language="en",
            temperature="0.2",
            delay_in_frames="16",
            keywords='{"words": ["LiteLLM"], "boost": 3}',
            target_language="fr",
        )

        assert _json_config(route) == {
            "language": "en",
            "temp": 0.2,
            "delay_in_frames": 16,
            "keywords": {"words": ["LiteLLM"], "boost": 3},
            "target_language": "fr",
        }


# Gradium tuning knobs (delay_in_frames, keywords, ...) only exist inside json_config.
def test_forwards_gradium_specific_settings_inside_json_config(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(url__startswith=_URL).mock(return_value=_HELLO_WORLD)

        _transcribe(language="en", delay_in_frames=16, keywords={"words": ["LiteLLM"], "boost": 3})

        assert _json_config(route) == {
            "language": "en",
            "delay_in_frames": 16,
            "keywords": {"words": ["LiteLLM"], "boost": 3},
        }


# Gradium documents its regional bases with an /api suffix, and pasting one must not double it.
def test_accepts_an_api_base_that_already_ends_in_api(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(url__startswith=_URL).mock(return_value=_HELLO_WORLD)

        litellm.transcription(
            model="gradium/default", file=("speech.wav", b"RIFF"), api_base=f"{_API_BASE}/api/", api_key=_API_KEY
        )

        assert route.called


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [("speech.ogg", "audio/ogg"), ("speech.opus", "audio/ogg"), ("speech.PCM", "audio/pcm"), ("speech", "audio/wav")],
)
def test_labels_each_supported_container_with_the_content_type_gradium_expects(
    gradium_env: None, filename: str, content_type: str
):
    with respx.mock:
        route: Final = respx.post(url__startswith=_URL).mock(return_value=_HELLO_WORLD)

        _transcribe(file=(filename, b"audio"))

        assert route.calls[-1].request.headers["content-type"] == content_type


# Gradium answers an unsupported container with a 500, which retries would only repeat.
@pytest.mark.parametrize("filename", ["speech.mp3", "speech.m4a", "speech.flac"])
def test_rejects_containers_gradium_cannot_decode_before_calling_it(gradium_env: None, filename: str):
    with respx.mock:
        route: Final = respx.post(url__startswith=_URL).mock(return_value=_HELLO_WORLD)

        with pytest.raises(litellm.BadRequestError, match="does not accept"):
            _transcribe(file=(filename, b"audio"))

        assert not route.called


# Gradium streams one ndjson message per segment, with no consolidated transcript.
def test_joins_the_streamed_segments_into_one_transcript_with_timestamps(gradium_env: None):
    with respx.mock:
        respx.post(url__startswith=_URL).mock(return_value=_HELLO_WORLD)

        response: Final = _transcribe()

        assert response.text == "Hello world."
        assert response["segments"] == [
            {"id": 0, "start": 0.5, "end": 1.0, "text": "Hello"},
            {"id": 1, "start": 1.2, "end": 2.5, "text": "world."},
        ]


# Gradium can fail after answering 200, as an error line inside the stream.
def test_raises_when_gradium_reports_an_error_mid_stream(gradium_env: None):
    with respx.mock:
        respx.post(url__startswith=_URL).mock(
            return_value=_ndjson(
                {"type": "text", "text": "Hello", "start_s": 0.5, "stream_id": 0},
                {"type": "error", "message": "Session is too long, please upgrade your plan. [1008]"},
            )
        )

        with pytest.raises(GradiumException, match="Session is too long") as raised:
            _transcribe()

        assert raised.value.status_code == 502


def test_surfaces_gradium_plain_text_errors(gradium_env: None):
    with respx.mock:
        respx.post(url__startswith=_URL).mock(
            return_value=httpx.Response(500, text="error from server 1008: API key is revoked or expired")
        )

        with pytest.raises(GradiumException, match="API key is revoked") as raised:
            _transcribe()

        assert raised.value.status_code == 500


def test_requires_an_api_key(gradium_env: None):
    with pytest.raises(litellm.AuthenticationError, match="API key is required"):
        litellm.transcription(model="gradium/default", file=("speech.wav", b"RIFF"), api_base=_API_BASE)


# Gradium bills 3 credits per started second of the stream it processed, which runs past the end of
# the file by the model's look-ahead (measured 2026-11-22: a 7.009 s WAV reported 12.08 s and cost 39 credits).
# Raw PCM has no header to measure, so the reported duration is the only way to bill it at all.
@pytest.mark.parametrize("file", [("speech.wav", _wav(seconds=2.0)), ("speech.pcm", b"\x00\x00" * _SAMPLE_RATE * 2)])
def test_bills_the_processed_duration_gradium_reports_rounded_up(
    gradium_env: None, local_model_cost_map: None, file: tuple[str, bytes]
):
    with respx.mock:
        respx.post(url__startswith=_URL).mock(return_value=_processed(0.08, 5.04))

        response: Final = _transcribe(file=file)

        price_per_second: Final = litellm.model_cost["gradium/default"]["input_cost_per_second"]
        assert price_per_second > 0
        assert response._hidden_params["response_cost"] == pytest.approx(6 * price_per_second)


def test_bills_the_file_duration_when_gradium_reports_no_processed_duration(
    gradium_env: None, local_model_cost_map: None
):
    with respx.mock:
        respx.post(url__startswith=_URL).mock(return_value=_HELLO_WORLD)

        response: Final = _transcribe(file=("speech.wav", _wav(seconds=2.0)))

        price_per_second: Final = litellm.model_cost["gradium/default"]["input_cost_per_second"]
        assert price_per_second > 0
        assert response._hidden_params["response_cost"] == pytest.approx(2.0 * price_per_second)


# gradium/default prices both endpoints, so adding the STT rate must leave TTS billed per character.
def test_speech_on_the_shared_default_model_still_bills_per_character(gradium_env: None, local_model_cost_map: None):
    with respx.mock:
        respx.post(f"{_API_BASE}/api/speech/tts").mock(return_value=httpx.Response(200, content=b"RIFFaudio"))

        response: Final = litellm.speech(
            model="gradium/default", input="a" * 100, voice="alloy", api_base=_API_BASE, api_key=_API_KEY
        )

        price_per_character: Final = litellm.model_cost["gradium/default"]["input_cost_per_character"]
        assert response._hidden_params["response_cost"] == pytest.approx(100 * price_per_character)


async def test_async_transcription_returns_the_same_transcript(gradium_env: None):
    with respx.mock:
        respx.post(url__startswith=_URL).mock(return_value=_HELLO_WORLD)

        response: Final = await litellm.atranscription(
            model="gradium/default", file=("speech.wav", b"RIFF"), api_base=_API_BASE, api_key=_API_KEY
        )

        assert response.text == "Hello world."
