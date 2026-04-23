import asyncio
import random
import os
import time

from .local_llm.chat_completion_proxy import get_chat_completion

HOST = os.getenv("OPENAI_API_BASE")
HOST_TYPE = os.getenv("BACKEND_TYPE")  # default None == ChatCompletion

import openai
from openai import OpenAI, AsyncOpenAI

# openai SDK v2.x moved configuration from module-level globals to client constructors.
# MemGPT historically uses OPENAI_API_BASE to point at OpenAI-compatible endpoints
# (LiteLLM proxy, Azure, local servers). We pass it through to the client constructor.
# An api_key is required by the v2 SDK even when the upstream proxy does not validate it.
_client_kwargs = {
    "api_key": os.getenv("OPENAI_API_KEY", "dummy"),
}
if HOST is not None:
    _client_kwargs["base_url"] = HOST

client = OpenAI(**_client_kwargs)
aclient = AsyncOpenAI(**_client_kwargs)


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
        return client.chat.completions.create(**kwargs)


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
        return await aclient.chat.completions.create(**kwargs)


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