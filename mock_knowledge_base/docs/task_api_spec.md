# Task API Specification

## Endpoints
- `GET /tasks` - List all tasks
- `POST /tasks` - Create new task
- `GET /tasks/{id}` - Get task details
- `PUT /tasks/{id}` - Update task
- `DELETE /tasks/{id}` - Delete task

## Data Model
```json
{
  "id": "string",
  "title": "string",
  "description": "string",
  "status": "todo|in_progress|done",
  "created_at": "datetime",
  "updated_at": "datetime"
}
``` 