"""LangChain integration for Azure AI Inference Plus with enhanced features."""

import json
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, Generation, LLMResult
from pydantic import ConfigDict, Field, model_validator

try:
    from azure_ai_inference_plus import AssistantMessage as AzureAssistantMessage
    from azure_ai_inference_plus import (  # Error classes for re-export; Schema format for structured output
        AzureAIInferencePlusError,
        AzureKeyCredential,
        ChatCompletionsClient,
        ConfigurationError,
        EmbeddingsClient,
        JsonSchemaFormat,
        JSONValidationError,
        RetryConfig,
        RetryExhaustedError,
    )
    from azure_ai_inference_plus import SystemMessage as AzureSystemMessage
    from azure_ai_inference_plus import UserMessage as AzureUserMessage
except ImportError as e:
    raise ImportError(
        "azure-ai-inference-plus is required for this integration. "
        "Install it with: pip install azure-ai-inference-plus"
    ) from e


class AzureAIInferencePlusChat(BaseChatModel):
    """LangChain Chat Model using Azure AI Inference Plus with enhanced features.

    Features:
    - Automatic reasoning separation for models like DeepSeek-R1
    - Built-in retry with exponential backoff
    - Guaranteed valid JSON with auto-retry
    - One import convenience
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(default=None, exclude=True)
    endpoint: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)
    model_name: str = Field(default="gpt-4")
    max_tokens: Optional[int] = Field(default=None)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=1.0)
    reasoning_tags: Optional[List[str]] = Field(default=None)
    response_format: Optional[str] = Field(default=None)
    retry_config: Optional[Any] = Field(
        default=None
    )  # RetryConfig type causes issues with mocking
    max_retries: int = Field(default=3)
    delay_seconds: float = Field(default=1.0)
    connection_timeout: Optional[float] = Field(default=None)

    def __init__(self, **data):
        """Initialize the chat model and set up the client."""
        super().__init__(**data)
        self._setup_client()

    def _setup_client(self):
        """Set up the Azure AI Inference Plus client."""
        # Create retry config if not provided
        if not self.retry_config:
            self.retry_config = RetryConfig(
                max_retries=self.max_retries,
                delay_seconds=self.delay_seconds,
            )

        # Initialize client
        if self.endpoint and self.api_key:
            client_kwargs = {
                "endpoint": self.endpoint,
                "credential": AzureKeyCredential(self.api_key),
                "retry_config": self.retry_config,
            }
            if self.connection_timeout is not None:
                client_kwargs["connection_timeout"] = self.connection_timeout
            self.client = ChatCompletionsClient(**client_kwargs)
        else:
            # Use environment variables
            client_kwargs = {"retry_config": self.retry_config}
            if self.connection_timeout is not None:
                client_kwargs["connection_timeout"] = self.connection_timeout
            self.client = ChatCompletionsClient(**client_kwargs)

    def _convert_langchain_to_azure_messages(
        self, messages: List[BaseMessage]
    ) -> List[Union[AzureSystemMessage, AzureUserMessage, AzureAssistantMessage]]:
        """Convert LangChain messages to Azure AI Inference Plus format."""
        azure_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                azure_messages.append(AzureSystemMessage(content=message.content))
            elif isinstance(message, HumanMessage):
                azure_messages.append(AzureUserMessage(content=message.content))
            elif isinstance(message, AIMessage):
                azure_messages.append(AzureAssistantMessage(content=message.content))
            else:
                # Default to user message for unknown types
                azure_messages.append(AzureUserMessage(content=str(message.content)))

        return azure_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using Azure AI Inference Plus."""

        # Convert messages
        azure_messages = self._convert_langchain_to_azure_messages(messages)

        # Prepare parameters
        params = {
            "messages": azure_messages,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        # Add optional parameters
        if self.reasoning_tags:
            params["reasoning_tags"] = self.reasoning_tags

        if self.response_format:
            params["response_format"] = self.response_format

        if stop:
            params["stop"] = stop

        # Override with any kwargs
        params.update(kwargs)

        try:
            # Make the request with automatic retry and JSON validation
            response = self.client.complete(**params)

            # Extract content and reasoning
            choice = response.choices[0]
            content = choice.message.content
            reasoning = getattr(choice.message, "reasoning", None)

            # Create additional kwargs for metadata
            additional_kwargs = {}
            if reasoning:
                additional_kwargs["reasoning"] = reasoning

            # Create AI message
            ai_message = AIMessage(content=content, additional_kwargs=additional_kwargs)

            # Create generation
            generation = ChatGeneration(
                message=ai_message,
                generation_info={
                    "finish_reason": choice.finish_reason,
                    "reasoning": reasoning,
                },
            )

            return ChatResult(generations=[generation])

        except Exception as e:
            raise ValueError(f"Error generating chat completion: {str(e)}")

    @property
    def _llm_type(self) -> str:
        """Return identifier of llm."""
        return "azure-ai-inference-plus-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_tags": self.reasoning_tags,
            "response_format": self.response_format,
        }


class AzureAIInferencePlusLLM(BaseLLM):
    """LangChain LLM using Azure AI Inference Plus for completion-style calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(default=None, exclude=True)
    endpoint: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)
    model_name: str = Field(default="gpt-4")
    max_tokens: Optional[int] = Field(default=None)
    temperature: float = Field(default=0.7)
    reasoning_tags: Optional[List[str]] = Field(default=None)
    response_format: Optional[str] = Field(default=None)
    retry_config: Optional[Any] = Field(default=None)
    connection_timeout: Optional[float] = Field(default=None)

    def __init__(self, **data):
        """Initialize the LLM and set up the client."""
        super().__init__(**data)
        self._setup_client()

    def _setup_client(self):
        """Set up the Azure AI Inference Plus client."""
        if not self.retry_config:
            self.retry_config = RetryConfig(max_retries=3, delay_seconds=1.0)

        if self.endpoint and self.api_key:
            client_kwargs = {
                "endpoint": self.endpoint,
                "credential": AzureKeyCredential(self.api_key),
                "retry_config": self.retry_config,
            }
            if self.connection_timeout is not None:
                client_kwargs["connection_timeout"] = self.connection_timeout
            self.client = ChatCompletionsClient(**client_kwargs)
        else:
            client_kwargs = {"retry_config": self.retry_config}
            if self.connection_timeout is not None:
                client_kwargs["connection_timeout"] = self.connection_timeout
            self.client = ChatCompletionsClient(**client_kwargs)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Azure AI Inference Plus API."""

        # Convert prompt to message format
        messages = [AzureUserMessage(content=prompt)]

        params = {
            "messages": messages,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.reasoning_tags:
            params["reasoning_tags"] = self.reasoning_tags

        if self.response_format:
            params["response_format"] = self.response_format

        if stop:
            params["stop"] = stop

        params.update(kwargs)

        try:
            response = self.client.complete(**params)
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error calling Azure AI Inference Plus: {str(e)}")

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate completions for the given prompts."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "azure-ai-inference-plus-llm"


class AzureAIInferencePlusEmbeddings(Embeddings):
    """LangChain Embeddings using Azure AI Inference Plus with enhanced features.

    Features:
    - Built-in retry with exponential backoff
    - Support for various embedding models
    - Batch processing for multiple texts
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-large",
        retry_config: Optional[Any] = None,
        max_retries: int = 3,
        delay_seconds: float = 1.0,
        connection_timeout: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the embeddings model and set up the client.

        Args:
            endpoint: Azure AI endpoint (uses env var if not provided)
            api_key: API key (uses env var if not provided)
            model_name: The embedding model to use
            retry_config: Custom retry configuration
            max_retries: Maximum number of retries
            delay_seconds: Initial delay between retries
            connection_timeout: Connection timeout in seconds
            **kwargs: Additional arguments
        """
        super().__init__()

        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.retry_config = retry_config
        self.max_retries = max_retries
        self.delay_seconds = delay_seconds
        self.connection_timeout = connection_timeout
        self.client = None

        self._setup_client()

    def _setup_client(self):
        """Set up the Azure AI Inference Plus embeddings client."""
        # Create retry config if not provided
        if not self.retry_config:
            self.retry_config = RetryConfig(
                max_retries=self.max_retries,
                delay_seconds=self.delay_seconds,
            )

        # Initialize embeddings client
        if (
            self.endpoint
            and self.api_key
            and isinstance(self.endpoint, str)
            and isinstance(self.api_key, str)
        ):
            client_kwargs = {
                "endpoint": self.endpoint,
                "credential": AzureKeyCredential(self.api_key),
                "retry_config": self.retry_config,
            }
            if self.connection_timeout is not None:
                client_kwargs["connection_timeout"] = self.connection_timeout
            self.client = EmbeddingsClient(**client_kwargs)
        else:
            # Use environment variables
            client_kwargs = {"retry_config": self.retry_config}
            if self.connection_timeout is not None:
                client_kwargs["connection_timeout"] = self.connection_timeout
            self.client = EmbeddingsClient(**client_kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, one for each input text
        """
        try:
            response = self.client.embed(input=texts, model=self.model_name)
            # Extract embeddings from response
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embed(input=[text], model=self.model_name)
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"Error generating query embedding: {str(e)}")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents asynchronously.

        Note: Currently falls back to synchronous implementation.
        """
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Generate embedding for query asynchronously.

        Note: Currently falls back to synchronous implementation.
        """
        return self.embed_query(text)


# Convenience functions for easy setup
def create_azure_chat_model(
    model_name: str = "gpt-4",
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    reasoning_tags: Optional[List[str]] = None,
    response_format: Optional[str] = None,
    connection_timeout: Optional[float] = None,
    **kwargs,
) -> AzureAIInferencePlusChat:
    """Create an Azure AI Inference Plus chat model with enhanced features.

    Args:
        model_name: The model to use (e.g., "DeepSeek-R1", "Codestral-2501")
        endpoint: Azure AI endpoint (uses env var if not provided)
        api_key: API key (uses env var if not provided)
        reasoning_tags: Tags for reasoning separation (e.g., ["<think>", "</think>"])
        response_format: "json_object" for guaranteed valid JSON
        connection_timeout: Connection timeout in seconds
        **kwargs: Additional parameters

    Returns:
        Configured AzureAIInferencePlusChat instance
    """
    return AzureAIInferencePlusChat(
        model_name=model_name,
        endpoint=endpoint,
        api_key=api_key,
        reasoning_tags=reasoning_tags,
        response_format=response_format,
        connection_timeout=connection_timeout,
        **kwargs,
    )


def create_azure_llm(
    model_name: str = "gpt-4",
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    connection_timeout: Optional[float] = None,
    **kwargs,
) -> AzureAIInferencePlusLLM:
    """Create an Azure AI Inference Plus LLM."""
    return AzureAIInferencePlusLLM(
        model_name=model_name,
        endpoint=endpoint,
        api_key=api_key,
        connection_timeout=connection_timeout,
        **kwargs,
    )


def create_azure_embeddings(
    model_name: str = "text-embedding-3-large",
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    connection_timeout: Optional[float] = None,
    **kwargs,
) -> AzureAIInferencePlusEmbeddings:
    """Create an Azure AI Inference Plus embeddings model with enhanced features.

    Args:
        model_name: The embedding model to use (e.g., "text-embedding-3-large", "text-embedding-ada-002")
        endpoint: Azure AI endpoint (uses env var if not provided)
        api_key: API key (uses env var if not provided)
        connection_timeout: Connection timeout in seconds
        **kwargs: Additional parameters

    Returns:
        Configured AzureAIInferencePlusEmbeddings instance
    """
    return AzureAIInferencePlusEmbeddings(
        model_name=model_name,
        endpoint=endpoint,
        api_key=api_key,
        connection_timeout=connection_timeout,
        **kwargs,
    )


# Export public API
__all__ = [
    # LangChain-specific classes and functions
    "AzureAIInferencePlusChat",
    "AzureAIInferencePlusLLM",
    "AzureAIInferencePlusEmbeddings",
    "create_azure_chat_model",
    "create_azure_llm",
    "create_azure_embeddings",
    # Re-exported from azure_ai_inference_plus for convenience
    "RetryConfig",
    "AzureKeyCredential",
    "JsonSchemaFormat",
    # Error classes for proper exception handling
    "AzureAIInferencePlusError",
    "JSONValidationError",
    "RetryExhaustedError",
    "ConfigurationError",
]
