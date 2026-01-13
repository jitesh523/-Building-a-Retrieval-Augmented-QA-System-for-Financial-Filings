
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag.generator import AnswerGenerator
from src.rag.pipeline import RAGPipeline

def test_generator():
    print("Testing AnswerGenerator...")
    try:
        # Use the key hardcoded in pipeline.py
        api_key = RAGPipeline.GEMINI_API_KEY
        print(f"Using API Key: {api_key[:10]}...")
        
        generator = AnswerGenerator(provider="gemini", api_key=api_key)
        
        print(f"Model: {generator.model}")
        print("Attempting to generate content...")
        
        # Test 1: Direct client usage
        response = generator.client.models.generate_content(
            model=generator.model,
            contents="Hello, are you working?"
        )
        print(f"✅ Direct Client Response: {response.text}")
        
        # Test 2: Generator method
        print("\nTesting generate() method...")
        mock_chunks = [{
            'chunk_text': 'Apple reported revenue of $100 billion.',
            'company': 'Apple',
            'dict': {},
            'similarity': 0.9
        }]
        
        answer = generator.generate("What is Apple's revenue?", mock_chunks)
        print(f"✅ Generator Answer: {answer.answer}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generator()
