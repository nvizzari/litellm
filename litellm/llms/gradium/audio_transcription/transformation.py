import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, ClassVar, Final, Literal
from urllib.parse import urlencode

from httpx import Headers, Response
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

import litellm
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes, TranscriptionResponse

from ..common_utils import GradiumException, gradium_api_url, gradium_auth_headers


class _Text(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["text"]
    text: str
    start_s: float
    stream_id: int = 0


class _EndText(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["end_text"]
    stop_s: float
    stream_id: int = 0


class _Error(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["error"]
    message: str


class _Untyped(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str


_MESSAGE: Final[TypeAdapter[_Text | _EndText | _Error]] = TypeAdapter(
    Annotated[_Text | _EndText | _Error, Field(discriminator="type")], config=ConfigDict(hide_input_in_errors=True)
)
_KNOWN_TYPES: Final = frozenset({"text", "end_text", "error"})


def _messages(body: str) -> tuple[_Text | _EndText | _Error, ...]:
    lines: Final = tuple(line for line in body.splitlines() if line.strip())
    return tuple(
        _MESSAGE.validate_json(line) for line in lines if _Untyped.model_validate_json(line).type in _KNOWN_TYPES
    )


class _Segment(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    start: float
    end: float
    text: str


def _segment_end(text: _Text, later: Sequence[_Text | _EndText | _Error]) -> float:
    return next(
        (message.stop_s for message in later if isinstance(message, _EndText) and message.stream_id == text.stream_id),
        text.start_s,
    )


def _segments(messages: Sequence[_Text | _EndText | _Error]) -> tuple[_Segment, ...]:
    texts: Final = tuple(
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, _Text) and message.text.strip()
    )
    return tuple(
        _Segment(
            id=segment_id,
            start=text.start_s,
            end=_segment_end(text, messages[index + 1 :]),
            text=text.text.strip(),
        )
        for segment_id, (index, text) in enumerate(texts)
    )


class GradiumAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    ASR_ENDPOINT_PATH = "/api/speech/asr"
    AUTO_DETECT_LANGUAGE = "any"
    NON_PROVIDER_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {"language", "temperature", "model", "OPENAI_TRANSCRIPTION_PARAMS"}
    )

    CONTENT_TYPES: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "wav": "audio/wav",
            "pcm": "audio/pcm",
            "ogg": "audio/ogg",
            "opus": "audio/ogg",
        }
    )

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: BaseAudioTranscriptionConfig declares list
        return ["language", "temperature"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: callers merge provider params into it downstream
        return {
            **optional_params,
            **{key: value for key, value in non_default_params.items() if key in ("language", "temperature")},
        }

    def _json_config(self, optional_params: Mapping[str, object]) -> Mapping[str, object]:
        language: Final = optional_params.get("language")
        temperature: Final = optional_params.get("temperature")
        provider_params: Final = {
            key: value
            for key, value in optional_params.items()
            if value is not None and key not in self.NON_PROVIDER_PARAMS
        }
        return {
            **provider_params,
            "language": language if isinstance(language, str) and language else self.AUTO_DETECT_LANGUAGE,
            **({"temp": temperature} if temperature is not None else {}),
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        litellm_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        stream: bool | None = None,
    ) -> str:
        query: Final = urlencode(
            {"model": model, "json_config": json.dumps(self._json_config(optional_params), separators=(",", ":"))}
        )
        return f"{gradium_api_url(api_base, model, self.ASR_ENDPOINT_PATH)}?{query}"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        litellm_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: the handler sets Content-Type on the returned headers
        return {**headers, **gradium_auth_headers(api_key, model)}

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | Headers,  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
    ) -> BaseLLMException:
        return GradiumException(message=error_message, status_code=status_code, headers=headers)

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        litellm_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
    ) -> AudioTranscriptionRequestData:
        audio: Final = process_audio_file(audio_file)
        extension: Final = audio.filename.rpartition(".")[2].lower() if "." in audio.filename else "wav"
        content_type: Final = self.CONTENT_TYPES.get(extension)
        if content_type is None:
            raise litellm.BadRequestError(
                message=(
                    f"Gradium speech-to-text does not accept .{extension} audio. "
                    f"Convert it to one of: {', '.join(f'.{name}' for name in self.CONTENT_TYPES)}."
                ),
                model=model,
                llm_provider="gradium",
            )
        return AudioTranscriptionRequestData(data=audio.file_content, content_type=content_type)

    def transform_audio_transcription_response(self, raw_response: Response) -> TranscriptionResponse:
        messages: Final = _messages(raw_response.text)
        error: Final = next((message for message in messages if isinstance(message, _Error)), None)
        if error is not None:
            raise GradiumException(message=error.message, status_code=502, headers=raw_response.headers)

        segments: Final = _segments(messages)
        return TranscriptionResponse(
            text=" ".join(segment.text for segment in segments),
            segments=[segment.model_dump() for segment in segments],
        )
