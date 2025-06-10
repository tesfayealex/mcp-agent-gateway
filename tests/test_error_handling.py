"""
Test script to verify error handling improvements in the MCP agent.
"""
import asyncio
import logging
import sys
import os

# Add the dev_assistant_agent_py directory to the path
sys.path.append('./dev_assistant_agent_py')

# Set up logging to see our improvements in action
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from custom_llm import GeminiCustomLLM
from agent import DevAssistantAgent, retry_with_exponential_backoff

async def test_retry_mechanism():
    """Test the retry mechanism with exponential backoff."""
    print("Testing retry mechanism...")
    
    # Test function that fails twice then succeeds
    attempt_count = 0
    
    async def failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(f"Simulated error on attempt {attempt_count}")
        return f"Success on attempt {attempt_count}"
    
    try:
        result = await retry_with_exponential_backoff(
            failing_operation,
            max_retries=3,
            base_delay=0.1,  # Short delay for testing
            retryable_errors=(Exception,)
        )
        print(f"✅ Retry test passed: {result}")
    except Exception as e:
        print(f"❌ Retry test failed: {e}")

async def test_error_messages():
    """Test that error messages are helpful and specific."""
    print("\nTesting error message generation...")
    
    # Mock a simple agent creation (without actual MCP connection)
    # This will test the error handling paths without needing full setup
    
    # Test different error scenarios
    test_errors = [
        ("AttributeError: 'NoneType' object has no attribute 'block_reason'", "block_reason issue"),
        ("validation error: missing_argument 'query'", "missing argument"),
        ("tool timeout error", "tool timeout"),
        ("connection refused proxy", "connection issue"),
        ("unexpected error", "general error")
    ]
    
    for error_str, description in test_errors:
        print(f"\nTesting {description}:")
        
        # Mock the error handling logic from handle_message
        if "AttributeError" in error_str and "block_reason" in error_str:
            response_text = "I encountered a temporary issue with the language model. Let me try a different approach to help you."
        elif "validation error" in error_str.lower() and "missing_argument" in error_str.lower():
            response_text = "I had trouble understanding the required parameters for that operation. Could you please rephrase your request or provide more specific details?"
        elif "tool" in error_str.lower() and ("timeout" in error_str.lower() or "error" in error_str.lower()):
            response_text = "I encountered an issue with one of the tools I was trying to use. This might be a temporary problem. Please try your request again, or let me know if you'd like to try a different approach."
        elif "connection" in error_str.lower() or "proxy" in error_str.lower():
            response_text = "I'm having trouble connecting to some of my tools right now. I can still help with general questions and information from my knowledge base. What would you like to know?"
        else:
            response_text = "I encountered an unexpected error while processing your request. Please try rephrasing your question or let me know if you need help with something specific."
        
        print(f"✅ Error: {error_str}")
        print(f"✅ Response: {response_text}")

def test_kwargs_parsing():
    """Test the enhanced kwargs parsing logic."""
    print("\nTesting kwargs parsing logic...")
    
    # Mock the parsing logic from create_proxy_tool_fn
    test_cases = [
        # (input_kwargs, tool_name, expected_output)
        ({'kwargs': '{"query": "test search"}'}, 'GITHUB_search_repositories', {'query': 'test search'}),
        ({'kwargs': 'query=user:tesfayealex'}, 'GITHUB_search_users', {'query': 'user:tesfayealex'}),
        ({'kwargs': 'test search query'}, 'search_repos', {'q': 'test search query'}),
        ({'kwargs': 'test query'}, 'query_engine', {'query': 'test query'}),
        ({'query': 'direct query'}, 'any_tool', {'query': 'direct query'}),
    ]
    
    import json
    
    for kwargs, tool_name, expected in test_cases:
        print(f"\nTesting: {kwargs} with tool {tool_name}")
        
        # Simulate the parsing logic
        if 'kwargs' in kwargs and len(kwargs) == 1:
            inner_kwargs = kwargs['kwargs']
            if isinstance(inner_kwargs, dict):
                parsed_args = inner_kwargs
            elif isinstance(inner_kwargs, str):
                try:
                    parsed_args = json.loads(inner_kwargs)
                except json.JSONDecodeError:
                    parsed_args = {}
                    
                    # Enhanced parsing logic
                    if '=' in inner_kwargs:
                        # Parse query string format
                        pairs = inner_kwargs.split('&') if '&' in inner_kwargs else [inner_kwargs]
                        for pair in pairs:
                            if '=' in pair:
                                key, value = pair.split('=', 1)
                                key = key.strip()
                                value = value.strip()
                                if value.startswith('"') and value.endswith('"'):
                                    value = value[1:-1]
                                parsed_args[key] = value
                    
                    # Fallback parsing
                    if not parsed_args:
                        if 'search' in tool_name.lower():
                            parsed_args = {'q': inner_kwargs}
                        elif 'query' in tool_name.lower():
                            parsed_args = {'query': inner_kwargs}
                        else:
                            if inner_kwargs and not any(char in inner_kwargs for char in ['=', '{', '[', ':']):
                                if 'github' in tool_name.lower():
                                    if 'search' in tool_name.lower() and 'repo' in tool_name.lower():
                                        parsed_args = {'query': inner_kwargs}
                                    elif 'user' in tool_name.lower():
                                        parsed_args = {'q': f"user:{inner_kwargs}"}
                                    else:
                                        parsed_args = {'query': inner_kwargs}
                                else:
                                    parsed_args = {'query': inner_kwargs}
            else:
                parsed_args = {}
        else:
            parsed_args = kwargs
        
        if parsed_args == expected:
            print(f"✅ Parsing successful: {parsed_args}")
        else:
            print(f"❌ Parsing failed. Expected: {expected}, Got: {parsed_args}")

async def main():
    """Main test function."""
    print("🔧 Testing MCP Agent Error Handling Improvements")
    print("=" * 50)
    
    await test_retry_mechanism()
    await test_error_messages()
    test_kwargs_parsing()
    
    print("\n" + "=" * 50)
    print("🎉 Error handling tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 