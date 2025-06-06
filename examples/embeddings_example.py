#!/usr/bin/env python3
"""
LangChain Embeddings example for Azure AI Inference Plus

This example demonstrates how to use the enhanced LangChain embeddings integration
with Azure AI Inference Plus, featuring automatic retry and LangChain compatibility.
"""

from dotenv import load_dotenv

from langchain_azure_ai_inference_plus import (
    AzureAIInferencePlusEmbeddings,
    AzureKeyCredential,
    RetryConfig,
    create_azure_embeddings,
)

# Load environment variables from .env file
load_dotenv()


def main():
    """Main example function for LangChain embeddings"""

    # Example 1: Basic LangChain embeddings
    print("=== Example 1: Basic LangChain Embeddings ===")

    try:
        # Create embeddings using convenience function - uses environment variables AZURE_AI_ENDPOINT and AZURE_AI_API_KEY
        embeddings = create_azure_embeddings(
            model_name="text-embedding-3-large"  # Replace with your embedding model
        )

        # Example documents to embed
        documents = [
            "LangChain is a framework for developing applications powered by language models",
            "Azure AI provides powerful embedding models for semantic search",
            "Vector databases enable similarity search over embeddings",
            "RAG (Retrieval Augmented Generation) combines retrieval and generation",
            "Embeddings capture semantic meaning in high-dimensional vectors",
        ]

        print(f"Generating embeddings for {len(documents)} documents...")

        # Generate embeddings for documents (batch processing)
        doc_embeddings = embeddings.embed_documents(documents)

        print(f"✅ Generated {len(doc_embeddings)} document embeddings")

        for i, (doc, embedding) in enumerate(zip(documents, doc_embeddings)):
            print(f"Document {i+1}: '{doc[:60]}{'...' if len(doc) > 60 else ''}'")
            print(f"  Embedding length: {len(embedding)}")
            print(f"  First 5 dimensions: {embedding[:5]}")
            print()

        # Generate embedding for a query
        query = "What is semantic search?"
        print(f"Generating embedding for query: '{query}'")

        query_embedding = embeddings.embed_query(query)
        print(f"✅ Query embedding length: {len(query_embedding)}")
        print(f"  First 5 dimensions: {query_embedding[:5]}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: LangChain Vector Store Integration
    print("\n=== Example 2: LangChain Vector Store Integration ===")

    try:
        # Demonstrate LangChain compatibility with a simple in-memory vector store
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        # Create embeddings model
        embeddings = create_azure_embeddings(model_name="text-embedding-3-large")

        # Create some sample documents
        docs = [
            Document(
                page_content="Python is a programming language",
                metadata={"source": "doc1"},
            ),
            Document(
                page_content="LangChain helps build LLM applications",
                metadata={"source": "doc2"},
            ),
            Document(
                page_content="Vector databases store embeddings",
                metadata={"source": "doc3"},
            ),
            Document(
                page_content="Semantic search finds similar content",
                metadata={"source": "doc4"},
            ),
        ]

        print(f"Creating FAISS vector store with {len(docs)} documents...")

        # Create vector store (this will call our embeddings.embed_documents)
        vector_store = FAISS.from_documents(docs, embeddings)

        print("✅ Vector store created successfully!")

        # Perform similarity search
        query = "programming language"
        similar_docs = vector_store.similarity_search(query, k=2)

        print(f"\nSimilarity search for: '{query}'")
        for i, doc in enumerate(similar_docs):
            print(
                f"Result {i+1}: '{doc.page_content}' (source: {doc.metadata['source']})"
            )

        # Perform similarity search with scores
        similar_docs_with_scores = vector_store.similarity_search_with_score(query, k=2)

        print(f"\nSimilarity search with scores:")
        for doc, score in similar_docs_with_scores:
            print(f"Score {score:.4f}: '{doc.page_content}'")

    except ImportError as e:
        if "faiss" in str(e).lower():
            print("⚠️  FAISS not installed. To run this example, install FAISS:")
            print("   For CPU-only: pip install faiss-cpu")
            print("   For GPU support: pip install faiss-gpu")
            print("")
            print("   💡 Tip: Our embeddings work with ANY LangChain vector store!")
            print("   Try: Chroma, Pinecone, Qdrant, Weaviate, etc.")
            print("")
            print("   ℹ️  Note: FAISS is not a core dependency due to its complex")
            print("       build requirements. We keep the core package lightweight!")
        else:
            print(f"⚠️  Missing dependency: {e}")
            print("   Install langchain-community: pip install langchain-community")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Custom retry configuration
    print("\n=== Example 3: Custom Retry Configuration ===")

    try:
        # Create embeddings with custom retry settings
        custom_embeddings = AzureAIInferencePlusEmbeddings(
            model_name="text-embedding-ada-002",  # Different model
            retry_config=RetryConfig(
                max_retries=5, delay_seconds=2.0, exponential_backoff=True
            ),
        )

        # Test with a single query
        test_query = "Custom retry configuration test"
        result = custom_embeddings.embed_query(test_query)

        print(f"✅ Custom retry embedding generated: {len(result)} dimensions")
        print(f"   Model: {custom_embeddings.model_name}")
        print(f"   Max retries: {custom_embeddings.retry_config.max_retries}")
        print(f"   Delay: {custom_embeddings.retry_config.delay_seconds}s")

    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Manual credential setup
    print("\n=== Example 4: Manual Credential Setup ===")

    try:
        # Manual embeddings setup (alternative to environment variables)
        manual_embeddings = AzureAIInferencePlusEmbeddings(
            endpoint="https://your-resource.services.ai.azure.com/models",  # Replace with your endpoint
            api_key="your-api-key-here",  # Replace with your API key
            model_name="text-embedding-3-large",
            max_retries=2,
            delay_seconds=1.5,
        )

        print(
            "✅ Embeddings model created with manual credentials (would work with real endpoint/key)"
        )
        print("✅ Alternative to environment variables for credential management")
        print(f"   Configured endpoint: {manual_embeddings.endpoint}")
        print(f"   Model: {manual_embeddings.model_name}")

    except Exception as e:
        print(
            f"Note: This example shows setup syntax (endpoint/key need to be real): {e}"
        )

    # Example 5: Async usage (preview)
    print("\n=== Example 5: Async Usage (Preview) ===")

    try:
        import asyncio

        async def async_embedding_example():
            embeddings = create_azure_embeddings()

            # Async methods (currently fallback to sync)
            query_result = await embeddings.aembed_query("Async test query")
            doc_results = await embeddings.aembed_documents(
                ["Async doc 1", "Async doc 2"]
            )

            print(f"✅ Async query embedding: {len(query_result)} dimensions")
            print(f"✅ Async document embeddings: {len(doc_results)} embeddings")
            print("   Note: Currently falls back to synchronous implementation")

        # Run async example
        asyncio.run(async_embedding_example())

    except Exception as e:
        print(f"Error in async example: {e}")

    print("\n=== Summary ===")
    print("✅ LangChain Azure AI Inference Plus Embeddings Features:")
    print("   🔄 Automatic retry with exponential backoff")
    print("   📦 Full LangChain BaseEmbeddings compatibility")
    print("   🏪 Works with all LangChain vector stores")
    print("   🔍 Perfect for RAG and semantic search applications")
    print("   ⚙️  Configurable retry behavior and logging")
    print("   🌐 Environment variable or manual credential support")


if __name__ == "__main__":
    main()
