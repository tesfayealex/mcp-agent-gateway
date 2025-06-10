import asyncio
from fastmcp import FastMCP
from fastmcp.tools.tool import Tool

# Create a simple tool to test schema preservation
def test_func(name: str, age: int = 25):
    '''Test function'''
    return f'Hello {name}, age {age}'

# Test creating tool directly with schema
custom_schema = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string', 'description': 'The person name'},
        'age': {'type': 'number', 'description': 'The person age', 'default': 25}
    },
    'required': ['name']
}

# Test 1: Create tool directly with custom schema
tool_direct = Tool(fn=test_func, name='test_tool_direct', description='Test tool direct', parameters=custom_schema)
print('Tool created directly with custom parameters:', tool_direct.parameters)

# Test 2: Create tool using from_function (what add_tool uses)
tool_from_function = Tool.from_function(fn=test_func, name='test_tool_from_function', description='Test tool from function')
print('Tool created from_function parameters:', tool_from_function.parameters)

# Test with FastMCP using direct tool registration
app = FastMCP('Test')

# Method 1: Try adding the direct tool with custom schema
if not hasattr(app, '_tools'):
    app._tools = {}
app._tools['test_tool_direct'] = tool_direct

# Method 2: Use add_tool (which uses from_function internally)
app.add_tool(test_func, name='test_tool_add_tool', description='Test tool add_tool')

async def test():
    tools = await app.get_tools()
    print(f'\nTools from get_tools: {len(tools)}')
    for tool_name, tool_obj in tools.items():
        print(f'\nTool: {tool_name}')
        print(f'Parameters: {tool_obj.parameters}')
        
        # Test to_mcp_tool method
        mcp_tool = tool_obj.to_mcp_tool()
        print(f'MCP tool inputSchema: {mcp_tool.inputSchema}')

asyncio.run(test()) 