# MCP Server Design

## Overview
This document describes the design of the MCP server that wraps the Task Management API.

## Architecture
- RESTful interface
- JWT authentication
- Rate limiting

## Implementation
See commit_def456 for initial implementation

## Open Questions
- Should we support WebSockets for real-time updates?
- What security model to use for agent communication? (See NEX-789) 