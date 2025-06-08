# langchain-azure-ai-inference-plus

**The easier way to use Azure AI Inference SDK with LangChain** ✨

[![PyPI Version](https://img.shields.io/pypi/v/langchain-azure-ai-inference-plus)](https://pypi.org/project/langchain-azure-ai-inference-plus/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/langchain-azure-ai-inference-plus)](https://pypi.org/project/langchain-azure-ai-inference-plus/)
[![License: MIT](https://img.shields.io/pypi/l/langchain-azure-ai-inference-plus)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Enhanced LangChain integration for [Azure AI Inference Plus](https://github.com/zpg6/azure-ai-inference-plus) with **automatic reasoning separation**, **guaranteed JSON validation**, and **smart retries**.

> **Note:** This package builds on [azure-ai-inference-plus](https://github.com/zpg6/azure-ai-inference-plus). For issues related to the underlying Azure AI functionality, please check there first before filing issues here.

## Why Use This Instead?

✅ **Reasoning separation** - clean output + accessible thinking (`.content` and `.additional_kwargs["reasoning"]`)  
✅ **Automatic retries** - never lose requests to transient failures  
✅ **JSON that works** - guaranteed valid JSON or automatic retry  
✅ **Full LangChain support** - works with chains, agents, tools, vector stores  
✅ **Embeddings included** - chat models + embeddings in one package  
✅ **One import** - no complex Azure SDK setup  
✅ **100% LangChain compatible** - drop-in replacement model for your current LangChain apps

## Installation

```bash
pip install langchain-azure-ai-inference-plus
```

Supports Python 3.10+

## Quick Start

```python
from langchain_azure_ai_inference_plus import create_azure_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# Uses environment variables: AZURE_AI_ENDPOINT, AZURE_AI_API_KEY
llm = create_azure_chat_model(
    model_name="Codestral-2501"
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

response = llm.invoke(messages)
print(response.content)
# "The capital of France is Paris..."
```

**Or with manual credentials (everything from one import!):**

```python
from langchain_azure_ai_inference_plus import create_azure_chat_model

llm = create_azure_chat_model(
    model_name="gpt-4",
    endpoint="https://your-resource.services.ai.azure.com/models",
    api_key="your-api-key"
)
```

## 🎯 Key Features

### 🧠 Automatic Reasoning Separation

**Game changer for reasoning models like DeepSeek-R1** - automatically separates thinking from output:

```python
llm = create_azure_chat_model(
    model_name="DeepSeek-R1",
    reasoning_tags=["<think>", "</think>"]  # ✨ Auto-separation
)

messages = [
    SystemMessage(content="You are a helpful math tutor."),
    HumanMessage(content="What's 15 * 23? Think step by step.")
]

result = llm.invoke(messages)

# Clean output without reasoning clutter
print(result.content)
# "15 * 23 equals 345."

# Access the reasoning separately
print(result.additional_kwargs.get("reasoning"))
# "Let me think about this step by step. 15 * 23 = 15 * 20 + 15 * 3..."
```

**For JSON mode, reasoning is automatically removed so you get clean JSON:**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

json_llm = create_azure_chat_model(
    model_name="DeepSeek-R1",
    reasoning_tags=["<think>", "</think>"],
    response_format="json_object"  # ✨ Clean JSON guaranteed
)

# Create a prompt template
json_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that returns JSON."),
    ("human", "Give me information about {city} in JSON format with keys: name, country, population, famous_landmarks")
])

# Create output parser
json_parser = JsonOutputParser()

# Chain them together
chain = json_prompt | json_llm | json_parser

# Execute with variable substitution
result = chain.invoke({"city": "Paris"})

# Pure JSON - reasoning automatically stripped
print(f"Parsed JSON result: {result}")
print(f"Population: {result.get('population', 'N/A')}")
```

### ✅ Guaranteed Valid JSON

No more JSON parsing errors - automatic validation and retry:

```python
json_llm = create_azure_chat_model(
    model_name="Codestral-2501",
    response_format="json_object"  # ✨ Auto-validation + retry
)

result = json_llm.invoke([
    HumanMessage(content="Give me a JSON response about Tokyo")
])

# Always valid JSON, no try/catch needed!
import json
data = json.loads(result.content)
```

### 🔄 Smart Automatic Retries

Built-in retry with exponential backoff - no configuration needed:

```python
# Automatically retries on failures - just works!
llm = create_azure_chat_model(model_name="Phi-4")
result = llm.invoke([HumanMessage(content="Tell me a joke")])
```

### ⚙️ Custom Retry Configuration

```python
from langchain_azure_ai_inference_plus import AzureAIInferencePlusChat
from azure_ai_inference_plus import RetryConfig

def custom_chat_retry(attempt, max_retries, exception, delay):
    print(f"🔄 Chat retry {attempt}/{max_retries}: {exception} (waiting {delay}s)")

def custom_json_retry(attempt, max_retries, message):
    print(f"📝 JSON retry {attempt}/{max_retries}: {message}")

# Create custom retry config
custom_retry_config = RetryConfig(
    max_retries=3,
    delay_seconds=1.0,
    exponential_backoff=True,
    on_chat_retry=custom_chat_retry,
    on_json_retry=custom_json_retry
)

llm = AzureAIInferencePlusChat(
    model_name="Phi-4",
    retry_config=custom_retry_config,
    connection_timeout=60.0  # Optional: connection timeout in seconds
)
```

### 🔗 LangChain Chains Integration

Works seamlessly with all LangChain components:

```python
from langchain_azure_ai_inference_plus import create_azure_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = create_azure_chat_model(
    model_name="Codestral-2501"
)

# Create a reusable prompt template
joke_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a witty programmer who tells short, clever jokes."),
    ("human", "Tell me a joke about {topic}")
])

# Chain with string output parser
joke_chain = joke_prompt | llm | StrOutputParser()

# Use the chain multiple times with different topics
joke = joke_chain.invoke({"topic": "programming"})
print(f"Programming joke: {joke}")
```

### 🚀 Embeddings Too

Full LangChain embeddings support with automatic retry:

```python
from langchain_azure_ai_inference_plus import create_azure_embeddings

embeddings = create_azure_embeddings(
    model_name="text-embedding-3-large"
)

# Example documents to embed
documents = [
    "LangChain is a framework for developing applications powered by language models",
    "Azure AI provides powerful embedding models for semantic search",
    "Vector databases enable similarity search over embeddings"
]

# Generate embeddings for documents (batch processing)
doc_embeddings = embeddings.embed_documents(documents)
print(f"Generated {len(doc_embeddings)} embeddings with {len(doc_embeddings[0])} dimensions")

# Generate embedding for a query
query = "What is semantic search?"
query_embedding = embeddings.embed_query(query)
print(f"Query embedding: {len(query_embedding)} dimensions")
```

**Works with any LangChain vector store:**

```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Create some sample documents
docs = [
    Document(page_content="Python is a programming language", metadata={"source": "doc1"}),
    Document(page_content="LangChain helps build LLM applications", metadata={"source": "doc2"}),
]

# Create vector store (automatically embeds documents)
vector_store = FAISS.from_documents(docs, embeddings)

# Perform similarity search
similar_docs = vector_store.similarity_search("programming language", k=1)
for doc in similar_docs:
    print(f"Found: {doc.page_content}")
```

## Environment Setup

Create a `.env` file:

```bash
AZURE_AI_ENDPOINT=https://your-resource.services.ai.azure.com/models
AZURE_AI_API_KEY=your-api-key-here
```

## Migration from Standard LangChain Azure Integration

**2 simple steps:**

1. `pip install langchain-azure-ai-inference-plus`
2. Change your import:

   ```python
   # Before
   from langchain_community.chat_models import AzureChatOpenAI

   # After
   from langchain_azure_ai_inference_plus import create_azure_chat_model

   # Create model (same interface, enhanced features)
   llm = create_azure_chat_model(model_name="gpt-4")
   ```

That's it! Your existing LangChain code works unchanged with automatic retries, JSON validation, and reasoning separation.

### Manual Credential Setup

```python
from langchain_azure_ai_inference_plus import create_azure_chat_model, create_azure_embeddings

# Chat model
llm = create_azure_chat_model(
    model_name="gpt-4",
    endpoint="https://your-resource.services.ai.azure.com/models",
    api_key="your-api-key"
)

# Embeddings
embeddings = create_azure_embeddings(
    model_name="text-embedding-3-large",
    endpoint="https://your-resource.services.ai.azure.com/models",
    api_key="your-api-key"
)
```

## Examples

Check out the [`examples/`](examples/) directory for complete demonstrations:

- [`basic_usage.py`](examples/basic_usage.py) - **Reasoning separation**, JSON validation, and LangChain chains
- [`embeddings_example.py`](examples/embeddings_example.py) - Embeddings with vector stores and retry features

All examples show real-world usage patterns with LangChain components.

## 🆚 Benefits Over Official LangChain Azure Integration [`langchain-azure-ai`](https://github.com/langchain-ai/langchain-azure)

| Feature                  | langchain-azure-ai             | LangChain Azure AI Inference Plus |
| ------------------------ | ------------------------------ | --------------------------------- |
| **Reasoning Separation** | ❌ Manual parsing required     | ✅ Automatic separation           |
| **JSON Validation**      | ❌ Manual try/catch needed     | ✅ Guaranteed valid JSON          |
| **Embeddings Support**   | ❌ Separate package required   | ✅ Unified chat + embeddings      |
| **Retry Logic**          | ❌ Manual implementation       | ✅ Built-in exponential backoff   |
| **Setup Complexity**     | ⚠️ Multi-step SDK setup        | ✅ One import, auto-config        |
| **Model Support**        | ✅ All Azure AI Foundry models | ✅ All Azure AI Foundry models    |
| **Observability**        | ❌ Limited retry visibility    | ✅ Optional retry callbacks       |

## License

[MIT](./LICENSE)

## Contributing

Contributions are welcome! Whether it's bug fixes, feature additions, or documentation improvements, we appreciate your help in making this project better. For major changes or new features, please open an issue first to discuss what you would like to change.

Made with ❤️ for the LangChain and Azure AI community
