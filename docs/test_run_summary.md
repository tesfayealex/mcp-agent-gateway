# Test Run Summary

**Run Date:** 2025-06-09 18:32:29
**Total Duration:** 1.09 seconds
**Total Tests:** 1
**Passed:** 1 | **Failed:** 0 | **Skipped:** 0

## Test Results

### ✅ test_proxy_tools_visibility
- **Status:** PASSED
- **Duration:** 0.01s
- **Details:** Found 50 tools

## Test Execution Logs

```
[18:32:30.290] INFO: === Starting test_proxy_tools_visibility ===
[18:32:30.296] INFO: All tools reported by proxy: ['list_managed_servers', 'GITHUB_add_issue_comment', 'GITHUB_add_pull_request_review_comment', 'GITHUB_create_branch', 'GITHUB_create_issue', 'GITHUB_create_or_update_file', 'GITHUB_create_pull_request', 'GITHUB_create_pull_request_review', 'GITHUB_create_repository', 'GITHUB_fork_repository', 'GITHUB_get_code_scanning_alert', 'GITHUB_get_commit', 'GITHUB_get_file_contents', 'GITHUB_get_issue', 'GITHUB_get_issue_comments', 'GITHUB_get_me', 'GITHUB_get_pull_request', 'GITHUB_get_pull_request_comments', 'GITHUB_get_pull_request_files', 'GITHUB_get_pull_request_reviews', 'GITHUB_get_pull_request_status', 'GITHUB_get_secret_scanning_alert', 'GITHUB_get_tag', 'GITHUB_list_branches', 'GITHUB_list_code_scanning_alerts', 'GITHUB_list_commits', 'GITHUB_list_issues', 'GITHUB_list_pull_requests', 'GITHUB_list_secret_scanning_alerts', 'GITHUB_list_tags', 'GITHUB_merge_pull_request', 'GITHUB_push_files', 'GITHUB_search_code', 'GITHUB_search_issues', 'GITHUB_search_repositories', 'GITHUB_search_users', 'GITHUB_update_issue', 'GITHUB_update_pull_request', 'GITHUB_update_pull_request_branch', 'filesystem_read_file', 'filesystem_read_multiple_files', 'filesystem_write_file', 'filesystem_edit_file', 'filesystem_create_directory', 'filesystem_list_directory', 'filesystem_directory_tree', 'filesystem_move_file', 'filesystem_search_files', 'filesystem_get_file_info', 'filesystem_list_allowed_directories']
[18:32:30.296] INFO: ✓ Found expected native tool: list_managed_servers
[18:32:30.296] WARNING: ✗ Missing expected native tool: get_server_tools
[18:32:30.296] INFO: ✓ Found expected GitHub tool: GITHUB_get_me
[18:32:30.296] INFO: ✓ Found expected filesystem tool: filesystem_list_directory
```
