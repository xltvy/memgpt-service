import typer
from llama_index.embeddings.openai import OpenAIEmbedding


def embedding_model():
    """Return a LlamaIndex embedding model configured per MemGPTConfig.

    Branches on config.embedding_provider ("openai" | "azure" | anything
    else -> huggingface). config.embedding_model holds the actual model
    identifier for that provider. config.embedding_endpoint_url, when set,
    routes the embedding path through a proxy — independent of the chat
    path, because they are independent concerns (a chat proxy may not
    expose an embeddings endpoint, and vice versa).
    """
    from memgpt.config import MemGPTConfig

    config = MemGPTConfig.load()
    provider = config.embedding_provider

    if provider == "openai":
        kwargs = {"model": config.embedding_model}
        # api_key: fall through to env var OPENAI_API_KEY by default; set
        # explicitly when config holds the key (LiteLLM master key, etc.)
        if config.openai_key:
            kwargs["api_key"] = config.openai_key
        # Embedding endpoint is independent of chat endpoint.
        if config.embedding_endpoint_url:
            kwargs["api_base"] = config.embedding_endpoint_url
        return OpenAIEmbedding(**kwargs)

    elif provider == "azure":
        return OpenAIEmbedding(
            model=config.embedding_model,
            deployment_name=config.azure_embedding_deployment,
            api_key=config.azure_key,
            api_base=config.azure_endpoint,
            api_type="azure",
            api_version=config.azure_version,
        )

    else:
        # HuggingFace path. Any provider string other than openai/azure
        # selects this; "huggingface" is the idiomatic value but "local"
        # also works (matches the archival/recall storage naming).
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        typer.secho(
            f"Loading HuggingFace embedding model {config.embedding_model}",
            fg=typer.colors.BLUE,
        )
        return HuggingFaceEmbedding(model_name=config.embedding_model)
