#!/usr/bin/env python3
"""
LangChain Extension Example for Azure AI Inference Plus

This example extends the basic azure-ai-inference-plus usage to show how the same
powerful features can be leveraged through LangChain's ecosystem:
- Same automatic retry with exponential backoff
- Same JSON validation and automatic retries
- Same reasoning separation for models like DeepSeek-R1
- PLUS: LangChain chains, prompts, output parsers, and more!
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_azure_ai_inference_plus import (
    AzureAIInferencePlusChat,
    AzureKeyCredential,
    RetryConfig,
    create_azure_chat_model,
)

# Load environment variables from .env file
load_dotenv()


def main():
    """Main example function showing LangChain integration"""

    # Example 1: Basic usage (same as azure-ai-inference-plus but through LangChain)
    print("=== Example 1: Basic Usage (LangChain Integration) ===")

    try:
        # Create LangChain-compatible model
        llm = create_azure_chat_model(model_name="Codestral-2501")

        # Use LangChain message format
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?"),
        ]

        response = llm.invoke(messages)
        print(f"Response: {response.content}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: JSON mode with reasoning models (enhanced with LangChain)
    print("\n=== Example 2: JSON Mode + Reasoning + LangChain Chains ===")

    try:
        # Create reasoning model with JSON output
        reasoning_llm = create_azure_chat_model(
            model_name="DeepSeek-R1",
            reasoning_tags=["<think>", "</think>"],
            response_format="json_object",
        )

        # Create a LangChain prompt template
        json_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that returns JSON."),
                (
                    "human",
                    "Give me information about {city} in JSON format with keys: name, country, population, famous_landmarks",
                ),
            ]
        )

        # Create output parser for automatic JSON parsing
        json_parser = JsonOutputParser()

        # Chain them together
        chain = json_prompt | reasoning_llm | json_parser

        # Execute the chain with variable substitution
        result = chain.invoke({"city": "Paris"})

        print(f"Parsed JSON result: {result}")
        print(f"Population: {result.get('population', 'N/A')}")
        print(f"Famous landmarks: {result.get('famous_landmarks', 'N/A')}")

        # Note: Reasoning is automatically extracted when available

    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Enhanced retry configuration (same config, LangChain integration)
    print("\n=== Example 3: Custom Retry + LangChain Prompt Templates ===")

    try:
        # Custom retry callbacks
        def custom_chat_retry(attempt, max_retries, exception, delay):
            print(f"Retry {attempt}/{max_retries}: {exception} (waiting {delay}s)")

        def custom_json_retry(attempt, max_retries, message):
            print(f"JSON retry {attempt}/{max_retries}: {message}")

        # Create custom retry config
        custom_retry_config = RetryConfig(
            max_retries=3,
            delay_seconds=1.0,
            exponential_backoff=True,
            on_chat_retry=custom_chat_retry,
            on_json_retry=custom_json_retry,
        )

        # Create LangChain model with custom retry
        custom_llm = AzureAIInferencePlusChat(
            model_name="Phi-4",
            retry_config=custom_retry_config,
        )

        # Create a reusable prompt template
        joke_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a witty programmer who tells short, clever jokes."),
                ("human", "Tell me a joke about {topic}"),
            ]
        )

        # Chain with string output parser
        joke_chain = joke_prompt | custom_llm | StrOutputParser()

        # Use the chain multiple times with different topics
        joke = joke_chain.invoke({"topic": "programming"})
        print(f"Programming joke: {joke}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Manual credential setup
    print("\n=== Example 4: Manual Credentials ===")

    try:
        # Manual client setup with credentials
        manual_llm = create_azure_chat_model(
            model_name="gpt-4",
            endpoint="https://your-resource.services.ai.azure.com/models",  # Replace with your endpoint
            api_key="your-api-key-here",  # Replace with your API key
        )

        print("✅ LangChain model created with manual credentials")

    except Exception as e:
        print(f"Note: Replace with your actual endpoint and API key: {e}")

    # Example 5: Advanced LangChain features showcase
    print("\n=== Example 5: Advanced LangChain Features Showcase ===")

    try:
        # Create model for advanced features
        advanced_llm = create_azure_chat_model(model_name="Codestral-2501")

        # Complex multi-step chain
        analysis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a business analyst. Analyze the given scenario step by step.",
                ),
                (
                    "human",
                    """
            Analyze this business scenario: {scenario}
            
            Please provide:
            1. Key challenges identified
            2. Potential solutions
            3. Risk assessment
            4. Recommended next steps
            
            Format your response clearly with numbered sections.
            """,
                ),
            ]
        )

        analysis_chain = analysis_prompt | advanced_llm | StrOutputParser()

        # Example business scenario
        scenario = "A small restaurant wants to expand to food delivery but has limited tech budget"

        analysis = analysis_chain.invoke({"scenario": scenario})
        print("Business Analysis Result:")
        print(analysis[:200] + "..." if len(analysis) > 200 else analysis)

    except Exception as e:
        print(f"Error in advanced example: {e}")


if __name__ == "__main__":
    main()
