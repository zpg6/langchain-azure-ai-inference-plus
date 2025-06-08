#!/usr/bin/env python3
"""
Tests for LangChain Azure AI Inference Plus integration.

These tests verify the LangChain integration functionality including
message conversion, reasoning separation, and JSON handling.
"""

import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_azure_ai_inference_plus import (
    AzureAIInferencePlusChat,
    AzureAIInferencePlusLLM,
    create_azure_chat_model,
    create_azure_llm,
)


def create_mock_chat_model(mock_client, **kwargs):
    """Helper function to create a mocked chat model."""
    with patch.object(AzureAIInferencePlusChat, "_setup_client"):
        model = AzureAIInferencePlusChat(**kwargs)
        model.client = mock_client
        return model


def create_mock_llm_model(mock_client, **kwargs):
    """Helper function to create a mocked LLM model."""
    with patch.object(AzureAIInferencePlusLLM, "_setup_client"):
        model = AzureAIInferencePlusLLM(**kwargs)
        model.client = mock_client
        return model


class TestMessageConversion:
    """Test LangChain to Azure AI message conversion"""

    def test_langchain_to_azure_message_conversion(self):
        """Test that LangChain messages are properly converted to Azure format"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        mock_client = Mock()
        chat_model = create_mock_chat_model(
            mock_client, endpoint=endpoint, api_key=api_key, model_name="gpt-4"
        )

        # Test different message types
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!"),
        ]

        azure_messages = chat_model._convert_langchain_to_azure_messages(messages)

        assert len(azure_messages) == 3
        # Messages should be converted without errors
        # The exact types are tested in the underlying library

    def test_unknown_message_type_conversion(self):
        """Test that unknown message types default to user messages"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        mock_client = Mock()
        chat_model = create_mock_chat_model(
            mock_client, endpoint=endpoint, api_key=api_key
        )

        # Create a custom message type
        class CustomMessage:
            def __init__(self, content):
                self.content = content

        messages = [CustomMessage("Custom message content")]
        azure_messages = chat_model._convert_langchain_to_azure_messages(messages)

        assert len(azure_messages) == 1


class TestReasoningIntegration:
    """Test reasoning separation in LangChain integration"""

    def test_reasoning_separation_in_langchain_response(self):
        """Test that reasoning is properly separated and accessible in LangChain response"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        # Create mock response with reasoning
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "The answer is 42."
        mock_message.reasoning = "Let me think about this step by step. The question asks for the meaning of life, and according to Douglas Adams, it's 42."
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.complete.return_value = mock_response

        chat_model = create_mock_chat_model(
            mock_client,
            endpoint=endpoint,
            api_key=api_key,
            model_name="DeepSeek-R1",
            reasoning_tags=["<think>", "</think>"],
        )

        messages = [HumanMessage(content="What's the meaning of life?")]
        result = chat_model._generate(messages)

        # Check that response is properly formatted
        generation = result.generations[0]
        assert generation.message.content == "The answer is 42."
        assert (
            generation.message.additional_kwargs["reasoning"]
            == "Let me think about this step by step. The question asks for the meaning of life, and according to Douglas Adams, it's 42."
        )
        assert (
            generation.generation_info["reasoning"]
            == "Let me think about this step by step. The question asks for the meaning of life, and according to Douglas Adams, it's 42."
        )

    def test_no_reasoning_in_response(self):
        """Test handling when no reasoning is present"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        # Create mock response without reasoning
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Hello! How can I help you?"
        mock_message.reasoning = None
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.complete.return_value = mock_response

        chat_model = create_mock_chat_model(
            mock_client, endpoint=endpoint, api_key=api_key, model_name="gpt-4"
        )

        messages = [HumanMessage(content="Hello")]
        result = chat_model._generate(messages)

        generation = result.generations[0]
        assert generation.message.content == "Hello! How can I help you?"
        assert "reasoning" not in generation.message.additional_kwargs
        assert generation.generation_info["reasoning"] is None


class TestJSONModeIntegration:
    """Test JSON mode functionality in LangChain integration"""

    def test_json_mode_response_format(self):
        """Test that JSON mode is properly passed to the underlying client"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        # Create mock JSON response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"result": "success", "data": {"value": 42}}'
        mock_message.reasoning = None
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.complete.return_value = mock_response

        chat_model = create_mock_chat_model(
            mock_client,
            endpoint=endpoint,
            api_key=api_key,
            model_name="Codestral-2501",
            response_format="json_object",
        )

        messages = [
            SystemMessage(content="You are a helpful assistant that returns JSON."),
            HumanMessage(content="Give me a JSON response"),
        ]
        result = chat_model._generate(messages)

        # Verify the client was called with JSON mode
        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args[1]
        assert call_args["response_format"] == "json_object"

        # Verify response is valid JSON
        generation = result.generations[0]
        json_data = json.loads(generation.message.content)
        assert json_data["result"] == "success"
        assert json_data["data"]["value"] == 42


class TestLLMIntegration:
    """Test LLM (non-chat) integration"""

    def test_llm_call_with_prompt(self):
        """Test LLM call with simple prompt"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        # Create mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "This is a response to your prompt."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.complete.return_value = mock_response

        llm = create_mock_llm_model(
            mock_client, endpoint=endpoint, api_key=api_key, model_name="gpt-4"
        )

        result = llm._call("Test prompt")

        # Verify the client was called correctly
        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args[1]
        assert len(call_args["messages"]) == 1
        assert call_args["model"] == "gpt-4"

        assert result == "This is a response to your prompt."


class TestRetryConfiguration:
    """Test retry configuration and callbacks"""

    def test_custom_retry_configuration(self):
        """Test custom retry configuration"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        with patch(
            "langchain_azure_ai_inference_plus.ChatCompletionsClient"
        ) as mock_client_class:
            mock_retry_config = Mock()
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            chat_model = AzureAIInferencePlusChat(
                endpoint=endpoint, api_key=api_key, retry_config=mock_retry_config
            )

            # Verify client was initialized with custom retry config
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args[1]
            assert call_args["retry_config"] == mock_retry_config


class TestConnectionTimeout:
    """Test connection timeout configuration"""

    def test_connection_timeout_passed_to_client(self):
        """Test that connection_timeout parameter is properly passed to underlying client"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"
        timeout_value = 60.0

        with patch(
            "langchain_azure_ai_inference_plus.ChatCompletionsClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            chat_model = AzureAIInferencePlusChat(
                endpoint=endpoint, 
                api_key=api_key, 
                connection_timeout=timeout_value
            )

            # Verify client was initialized with connection timeout
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args[1]
            assert call_args["connection_timeout"] == timeout_value

    def test_connection_timeout_with_env_credentials(self):
        """Test connection timeout with environment variable credentials"""
        timeout_value = 45.0

        with patch(
            "langchain_azure_ai_inference_plus.ChatCompletionsClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            chat_model = AzureAIInferencePlusChat(
                connection_timeout=timeout_value
            )

            # Verify client was initialized with connection timeout
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args[1]
            assert call_args["connection_timeout"] == timeout_value

    def test_no_connection_timeout_parameter(self):
        """Test that connection_timeout is not passed when not specified"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        with patch(
            "langchain_azure_ai_inference_plus.ChatCompletionsClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            chat_model = AzureAIInferencePlusChat(
                endpoint=endpoint, 
                api_key=api_key
            )

            # Verify client was initialized without connection timeout
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args[1]
            assert "connection_timeout" not in call_args

    def test_create_azure_chat_model_with_connection_timeout(self):
        """Test convenience function with connection timeout"""
        timeout_value = 120.0

        with patch("langchain_azure_ai_inference_plus.ChatCompletionsClient"):
            with patch("langchain_azure_ai_inference_plus.RetryConfig"):
                chat_model = create_azure_chat_model(
                    model_name="Codestral-2501",
                    connection_timeout=timeout_value
                )

                assert isinstance(chat_model, AzureAIInferencePlusChat)
                assert chat_model.connection_timeout == timeout_value


class TestConvenienceFunctions:
    """Test convenience functions for creating models"""

    def test_create_azure_chat_model_with_reasoning(self):
        """Test creating chat model with reasoning configuration"""
        with patch("langchain_azure_ai_inference_plus.ChatCompletionsClient"):
            with patch("langchain_azure_ai_inference_plus.RetryConfig"):
                chat_model = create_azure_chat_model(
                    model_name="DeepSeek-R1",
                    reasoning_tags=["<think>", "</think>"],
                )

                assert isinstance(chat_model, AzureAIInferencePlusChat)
                assert chat_model.model_name == "DeepSeek-R1"
                assert chat_model.reasoning_tags == ["<think>", "</think>"]

    def test_create_azure_chat_model_with_json_mode(self):
        """Test creating chat model with JSON mode"""
        with patch("langchain_azure_ai_inference_plus.ChatCompletionsClient"):
            with patch("langchain_azure_ai_inference_plus.RetryConfig"):
                chat_model = create_azure_chat_model(
                    model_name="Codestral-2501", response_format="json_object"
                )

                assert isinstance(chat_model, AzureAIInferencePlusChat)
                assert chat_model.response_format == "json_object"

    def test_create_azure_llm_basic(self):
        """Test creating basic LLM"""
        with patch("langchain_azure_ai_inference_plus.ChatCompletionsClient"):
            with patch("langchain_azure_ai_inference_plus.RetryConfig"):
                llm = create_azure_llm(model_name="gpt-4", temperature=0.8)

                assert isinstance(llm, AzureAIInferencePlusLLM)
                assert llm.model_name == "gpt-4"
                assert llm.temperature == 0.8


class TestModelProperties:
    """Test model properties and identifiers"""

    def test_chat_model_llm_type(self):
        """Test chat model LLM type identifier"""
        with patch("langchain_azure_ai_inference_plus.ChatCompletionsClient"):
            with patch("langchain_azure_ai_inference_plus.RetryConfig"):
                chat_model = AzureAIInferencePlusChat(
                    endpoint="https://test.example.com", api_key="test-key"
                )
                assert chat_model._llm_type == "azure-ai-inference-plus-chat"

    def test_llm_model_llm_type(self):
        """Test LLM model LLM type identifier"""
        with patch("langchain_azure_ai_inference_plus.ChatCompletionsClient"):
            with patch("langchain_azure_ai_inference_plus.RetryConfig"):
                llm = AzureAIInferencePlusLLM(
                    endpoint="https://test.example.com", api_key="test-key"
                )
                assert llm._llm_type == "azure-ai-inference-plus-llm"

    def test_identifying_params(self):
        """Test identifying parameters for model comparison"""
        with patch("langchain_azure_ai_inference_plus.ChatCompletionsClient"):
            with patch("langchain_azure_ai_inference_plus.RetryConfig"):
                chat_model = AzureAIInferencePlusChat(
                    endpoint="https://test.example.com",
                    api_key="test-key",
                    model_name="test-model",
                    temperature=0.5,
                    max_tokens=100,
                    reasoning_tags=["<think>", "</think>"],
                    response_format="json_object",
                )

                params = chat_model._identifying_params
                assert params["model_name"] == "test-model"
                assert params["temperature"] == 0.5
                assert params["max_tokens"] == 100
                assert params["reasoning_tags"] == ["<think>", "</think>"]
                assert params["response_format"] == "json_object"


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_generation_error_handling(self):
        """Test that generation errors are properly wrapped"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        mock_client = Mock()
        mock_client.complete.side_effect = Exception("API Error")

        chat_model = create_mock_chat_model(
            mock_client, endpoint=endpoint, api_key=api_key
        )

        messages = [HumanMessage(content="Test")]

        with pytest.raises(ValueError, match="Error generating chat completion"):
            chat_model._generate(messages)

    def test_llm_call_error_handling(self):
        """Test that LLM call errors are properly wrapped"""
        endpoint = "https://test.openai.azure.com"
        api_key = "test-key"

        mock_client = Mock()
        mock_client.complete.side_effect = Exception("API Error")

        llm = create_mock_llm_model(mock_client, endpoint=endpoint, api_key=api_key)

        with pytest.raises(ValueError, match="Error calling Azure AI Inference Plus"):
            llm._call("Test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
