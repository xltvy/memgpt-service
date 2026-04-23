import asyncio
import random
import os
import time

from .local_llm.chat_completion_proxy import get_chat_completion

HOST_TYPE = os.getenv("BACKEND_TYPE")  # default None == ChatCompletion
import openai
from openai import OpenAI, AsyncOpenAI

from memgpt.openai_compat import translate_request, translate_response


# Credential resolution: env overrides config; config is the authoritative
# source of deployment topology. Env precedence keeps dev ergonomics (A/B a
# different endpoint in a single shell, no file edits) while config makes
# the project installable from file alone — prerequisite for shipping as an
# OpenClaw plugin, where env-driven setup is unacceptable.
#
# Resolution is lazy: memgpt configure imports this module before any config
# file exists, so construction-time access to MemGPTConfig would crash the
# wizard. The proxy below defers resolution until first real attribute
# access (i.e. the first actual chat/embeddings call).
#
# Failure is loud: api.openai.com with a dummy key is a silent-fallback
# footgun this project cannot afford — experimental runs would look
# "successful" while contacting an unintended provider, corrupting the
# provenance chain on which a memory-security study depends.
def _resolve_openai_client_kwargs():
    from memgpt.config import MemGPTConfig
    cfg = MemGPTConfig.load() if MemGPTConfig.exists() else None

    api_key = os.getenv("OPENAI_API_KEY") or (cfg.openai_key if cfg else None)
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key found. Set OPENAI_API_KEY in the environment "
            "or [openai] key = ... in ~/.memgpt/config. If routing chat "
            "through LiteLLM, this should be the LiteLLM master key, not "
            "the upstream provider key."
        )

    base_url = os.getenv("OPENAI_API_BASE") or (cfg.openai_endpoint_url if cfg else None)

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    # Provenance log: every run records the endpoint that served it.
    print(f"[memgpt] openai client base_url={base_url or '<sdk default: api.openai.com>'}")
    return kwargs


class _LazyOpenAIClient:
    """Defers OpenAI client construction to first real attribute access.
    Call sites (client.chat.completions.create, client.embeddings.create,
    etc.) remain unchanged."""
    def __init__(self, cls):
        self._cls = cls
        self._instance = None

    def _ensure(self):
        if self._instance is None:
            self._instance = self._cls(**_resolve_openai_client_kwargs())
        return self._instance

    def __getattr__(self, name):
        return getattr(self._ensure(), name)


client = _LazyOpenAIClient(OpenAI)
aclient = _LazyOpenAIClient(AsyncOpenAI)

# Back-compat: some call sites may still reference HOST for logging.
HOST = os.getenv("OPENAI_API_BASE")


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 20,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    # Local model
    if HOST_TYPE is not None:
        return get_chat_completion(**kwargs)

    # OpenAI / Azure model
    else:
        if using_azure():
            azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if azure_openai_deployment is not None:
                kwargs["deployment_id"] = azure_openai_deployment
            else:
                kwargs["engine"] = MODEL_TO_AZURE_ENGINE[kwargs["model"]]
                kwargs.pop("model")
        translated_kwargs = translate_request(kwargs)
        response = client.chat.completions.create(**translated_kwargs)
        return translate_response(response)


def aretry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 20,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    async def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return await func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                print(f"acreate (backoff): caught error: {e}")
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                await asyncio.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@aretry_with_exponential_backoff
async def acompletions_with_backoff(**kwargs):
    # Local model
    if HOST_TYPE is not None:
        return get_chat_completion(**kwargs)

    # OpenAI / Azure model
    else:
        if using_azure():
            azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if azure_openai_deployment is not None:
                kwargs["deployment_id"] = azure_openai_deployment
            else:
                kwargs["engine"] = MODEL_TO_AZURE_ENGINE[kwargs["model"]]
                kwargs.pop("model")
        translated_kwargs = translate_request(kwargs)
        response = await aclient.chat.completions.create(**translated_kwargs)
        return translate_response(response)


@aretry_with_exponential_backoff
async def acreate_embedding_with_backoff(**kwargs):
    """Wrapper around Embedding.acreate w/ backoff"""
    if using_azure():
        azure_openai_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        if azure_openai_deployment is not None:
            kwargs["deployment_id"] = azure_openai_deployment
        else:
            kwargs["engine"] = kwargs["model"]
            kwargs.pop("model")
    return await aclient.embeddings.create(**kwargs)


async def async_get_embedding_with_backoff(text, model="text-embedding-ada-002"):
    """To get text embeddings, import/call this function
    It specifies defaults + handles rate-limiting + is async"""
    text = text.replace("\n", " ")
    response = await acreate_embedding_with_backoff(input=[text], model=model)
    # openai v2.x returns pydantic model objects; access via attribute, not dict
    embedding = response.data[0].embedding
    return embedding


@retry_with_exponential_backoff
def create_embedding_with_backoff(**kwargs):
    if using_azure():
        azure_openai_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        if azure_openai_deployment is not None:
            kwargs["deployment_id"] = azure_openai_deployment
        else:
            kwargs["engine"] = kwargs["model"]
            kwargs.pop("model")
    return client.embeddings.create(**kwargs)


def get_embedding_with_backoff(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = create_embedding_with_backoff(input=[text], model=model)
    # openai v2.x returns pydantic model objects; access via attribute, not dict
    embedding = response.data[0].embedding
    return embedding


MODEL_TO_AZURE_ENGINE = {
    "gpt-4": "gpt-4",
    "gpt-4-32k": "gpt-4-32k",
    "gpt-3.5": "gpt-35-turbo",
    "gpt-3.5-turbo": "gpt-35-turbo",
    "gpt-3.5-turbo-16k": "gpt-35-turbo-16k",
}


def get_set_azure_env_vars():
    azure_env_variables = [
        ("AZURE_OPENAI_KEY", os.getenv("AZURE_OPENAI_KEY")),
        ("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")),
        ("AZURE_OPENAI_VERSION", os.getenv("AZURE_OPENAI_VERSION")),
        ("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT")),
        (
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
            os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        ),
    ]
    return [x for x in azure_env_variables if x[1] is not None]


def using_azure():
    return len(get_set_azure_env_vars()) > 0


def configure_azure_support():
    # NOTE: openai v2.x removed module-level globals (openai.api_type etc.).
    # A proper Azure port requires the AzureOpenAI() client.
    # Azure is not used in Persival's setup; this function is a no-op stub
    # that warns rather than silently setting unused module attributes.
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_version = os.getenv("AZURE_OPENAI_VERSION")
    if None in [
        azure_openai_key,
        azure_openai_endpoint,
        azure_openai_version,
    ]:
        print(f"Error: missing Azure OpenAI environment variables. Please see README section on Azure.")
        return

    print(
        "Warning: configure_azure_support() is a no-op in the openai v2.x port. "
        "Azure users must migrate to AzureOpenAI() client; not implemented in this fork."
    )


def check_azure_embeddings():
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    if azure_openai_deployment is not None and azure_openai_embedding_deployment is None:
        raise ValueError(
            f"Error: It looks like you are using Azure deployment ids and computing embeddings, make sure you are setting one for embeddings as well. Please see README section on Azure"
        )