#!/usr/bin/env python3
"""
Tests for AzureAIInferencePlusEmbeddings

These tests verify the LangChain Azure AI Inference Plus embeddings functionality.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from langchain_azure_ai_inference_plus import (
    AzureAIInferencePlusEmbeddings,
    AzureKeyCredential,
    RetryConfig,
    create_azure_embeddings,
)


class TestAzureAIInferencePlusEmbeddings:
    """Test the enhanced LangChain embeddings integration"""

    def test_init_with_params(self):
        """Test embeddings initialization with explicit parameters"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        embeddings = AzureAIInferencePlusEmbeddings(
            endpoint=endpoint,
            api_key=api_key,
            model_name="text-embedding-3-large",
            max_retries=5,
            delay_seconds=2.0,
        )

        assert embeddings.endpoint == endpoint
        assert embeddings.api_key == api_key
        assert embeddings.model_name == "text-embedding-3-large"
        assert embeddings.max_retries == 5
        assert embeddings.delay_seconds == 2.0
        assert embeddings.client is not None

    def test_init_with_env_vars(self):
        """Test embeddings initialization with environment variables"""
        with patch.dict(
            os.environ,
            {
                "AZURE_AI_ENDPOINT": "https://test.openai.azure.com",
                "AZURE_AI_API_KEY": "test-key",
            },
        ):
            embeddings = AzureAIInferencePlusEmbeddings()
            assert embeddings.client is not None
            assert embeddings.model_name == "text-embedding-3-large"  # Default

    @patch("langchain_azure_ai_inference_plus.EmbeddingsClient")
    def test_convenience_function(self, mock_embeddings_client):
        """Test the create_azure_embeddings convenience function"""
        mock_client_instance = MagicMock()
        mock_embeddings_client.return_value = mock_client_instance

        embeddings = create_azure_embeddings(model_name="text-embedding-ada-002")

        assert isinstance(embeddings, AzureAIInferencePlusEmbeddings)
        assert embeddings.model_name == "text-embedding-ada-002"

    @patch("langchain_azure_ai_inference_plus.EmbeddingsClient")
    def test_embed_documents(self, mock_embeddings_client):
        """Test embedding multiple documents"""
        # Mock the client and response
        mock_client_instance = MagicMock()
        mock_embeddings_client.return_value = mock_client_instance

        # Mock response structure
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client_instance.embed.return_value = mock_response

        embeddings = AzureAIInferencePlusEmbeddings()
        texts = ["Hello world", "Python is great"]

        result = embeddings.embed_documents(texts)

        # Verify the client was called correctly
        mock_client_instance.embed.assert_called_once_with(
            input=texts, model="text-embedding-3-large"
        )

        # Verify the result
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch("langchain_azure_ai_inference_plus.EmbeddingsClient")
    def test_embed_query(self, mock_embeddings_client):
        """Test embedding a single query"""
        # Mock the client and response
        mock_client_instance = MagicMock()
        mock_embeddings_client.return_value = mock_client_instance

        # Mock response structure
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.7, 0.8, 0.9])]
        mock_client_instance.embed.return_value = mock_response

        embeddings = AzureAIInferencePlusEmbeddings()
        query = "What is Python?"

        result = embeddings.embed_query(query)

        # Verify the client was called correctly
        mock_client_instance.embed.assert_called_once_with(
            input=[query], model="text-embedding-3-large"
        )

        # Verify the result
        assert result == [0.7, 0.8, 0.9]

    @patch("langchain_azure_ai_inference_plus.EmbeddingsClient")
    def test_embed_documents_error_handling(self, mock_embeddings_client):
        """Test error handling in embed_documents"""
        # Mock the client to raise an exception
        mock_client_instance = MagicMock()
        mock_embeddings_client.return_value = mock_client_instance
        mock_client_instance.embed.side_effect = Exception("API Error")

        embeddings = AzureAIInferencePlusEmbeddings()

        with pytest.raises(ValueError, match="Error generating embeddings: API Error"):
            embeddings.embed_documents(["test"])

    @patch("langchain_azure_ai_inference_plus.EmbeddingsClient")
    def test_embed_query_error_handling(self, mock_embeddings_client):
        """Test error handling in embed_query"""
        # Mock the client to raise an exception
        mock_client_instance = MagicMock()
        mock_embeddings_client.return_value = mock_client_instance
        mock_client_instance.embed.side_effect = Exception("API Error")

        embeddings = AzureAIInferencePlusEmbeddings()

        with pytest.raises(
            ValueError, match="Error generating query embedding: API Error"
        ):
            embeddings.embed_query("test")

    @patch("langchain_azure_ai_inference_plus.EmbeddingsClient")
    def test_retry_config_creation(self, mock_embeddings_client):
        """Test that retry config is properly created"""
        mock_client_instance = MagicMock()
        mock_embeddings_client.return_value = mock_client_instance

        embeddings = AzureAIInferencePlusEmbeddings(max_retries=10, delay_seconds=3.0)

        assert embeddings.retry_config is not None
        assert embeddings.retry_config.max_retries == 10
        assert embeddings.retry_config.delay_seconds == 3.0

    @patch("langchain_azure_ai_inference_plus.EmbeddingsClient")
    def test_connection_timeout_passed_to_client(self, mock_embeddings_client):
        """Test that connection_timeout parameter is properly passed to embeddings client"""
        mock_client_instance = MagicMock()
        mock_embeddings_client.return_value = mock_client_instance

        timeout_value = 30.0
        embeddings = AzureAIInferencePlusEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            connection_timeout=timeout_value
        )

        # Verify client was initialized with connection timeout
        mock_embeddings_client.assert_called_once()
        call_args = mock_embeddings_client.call_args[1]
        assert call_args["connection_timeout"] == timeout_value

    @patch("langchain_azure_ai_inference_plus.EmbeddingsClient")
    def test_create_azure_embeddings_with_connection_timeout(self, mock_embeddings_client):
        """Test convenience function with connection timeout"""
        mock_client_instance = MagicMock()
        mock_embeddings_client.return_value = mock_client_instance

        timeout_value = 45.0
        embeddings = create_azure_embeddings(
            model_name="text-embedding-ada-002",
            connection_timeout=timeout_value
        )

        assert isinstance(embeddings, AzureAIInferencePlusEmbeddings)
        assert embeddings.connection_timeout == timeout_value

    @pytest.mark.asyncio
    @patch("langchain_azure_ai_inference_plus.EmbeddingsClient")
    async def test_async_embed_documents(self, mock_embeddings_client):
        """Test async embed_documents (falls back to sync)"""
        # Mock the client and response
        mock_client_instance = MagicMock()
        mock_embeddings_client.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client_instance.embed.return_value = mock_response

        embeddings = AzureAIInferencePlusEmbeddings()

        result = await embeddings.aembed_documents(["test"])

        assert result == [[0.1, 0.2, 0.3]]

    @pytest.mark.asyncio
    @patch("langchain_azure_ai_inference_plus.EmbeddingsClient")
    async def test_async_embed_query(self, mock_embeddings_client):
        """Test async embed_query (falls back to sync)"""
        # Mock the client and response
        mock_client_instance = MagicMock()
        mock_embeddings_client.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client_instance.embed.return_value = mock_response

        embeddings = AzureAIInferencePlusEmbeddings()

        result = await embeddings.aembed_query("test")

        assert result == [0.1, 0.2, 0.3]


if __name__ == "__main__":
    pytest.main([__file__])
