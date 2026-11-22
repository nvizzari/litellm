from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException


class GradiumException(BaseLLMException):
    pass


def gradium_api_url(api_base: str | None, model: str, path: str) -> str:
    if not api_base:
        raise litellm.APIConnectionError(
            message="Gradium API base is required. Set `api_base` on the model, e.g. https://eu.api.gradium.ai.",
            llm_provider="gradium",
            model=model,
        )

    root: Final = api_base.rstrip("/").removesuffix("/api")
    return f"{root}{path}"


def gradium_auth_headers(api_key: str | None, model: str) -> Mapping[str, str]:
    if not api_key:
        raise litellm.AuthenticationError(
            message="Gradium API key is required. Set `api_key` on the model.",
            llm_provider="gradium",
            model=model,
        )

    return MappingProxyType({"x-api-key": api_key})
