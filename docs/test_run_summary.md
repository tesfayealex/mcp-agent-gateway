# Test Run Summary

**Run Date:** 2025-06-01 11:11:34
**Total Duration:** 2.55 seconds
**Total Tests:** 5
**Passed:** 2 | **Failed:** 1 | **Skipped:** 2

## Test Results

### ✅ test_proxy_tools_visibility
- **Status:** PASSED
- **Duration:** 0.00s
- **Details:** Found 3 tools

### ❌ test_native_proxy_tools
- **Status:** FAILED
- **Duration:** 0.01s
- **Details:** No connected and enabled server found for testing
assert None is not None

### ⏭️ test_github_tools
- **Status:** SKIPPED
- **Duration:** 0.01s
- **Details:** GitHub tool 'GITHUB_get_me' not found

### ⏭️ test_filesystem_tools
- **Status:** SKIPPED
- **Duration:** 0.01s
- **Details:** Filesystem tool 'filesystem_list_directory' not found

### ✅ test_tool_call_integration
- **Status:** PASSED
- **Duration:** 0.27s
- **Details:** Integration test completed successfully

## Test Execution Logs

```
[11:11:35.471] INFO: === Starting test_proxy_tools_visibility ===
[11:11:35.476] INFO: All tools reported by proxy: ['list_managed_servers', 'get_server_tools', 'call_server_tool']
[11:11:35.476] INFO: ✓ Found expected native tool: list_managed_servers
[11:11:35.476] INFO: ✓ Found expected native tool: get_server_tools
[11:11:35.476] WARNING: ✗ Missing expected GitHub tool: GITHUB_get_me
[11:11:35.476] WARNING: ✗ Missing expected filesystem tool: filesystem_list_directory
[11:11:35.923] INFO: === Starting test_native_proxy_tools ===
[11:11:35.923] INFO: Testing 'list_managed_servers'...
[11:11:35.927] INFO: Found 2 managed servers
[11:11:35.928] ERROR: Test test_native_proxy_tools failed: No connected and enabled server found for testing
assert None is not None
[11:11:36.352] INFO: === Starting test_github_tools ===
[11:11:36.795] INFO: === Starting test_filesystem_tools ===
[11:11:37.203] INFO: === Starting test_tool_call_integration ===
[11:11:37.469] INFO: Server 'GITHUB' has 38 tools
[11:11:37.469] WARNING: ✗ Proxied tool missing: GITHUB_add_issue_comment
[11:11:37.469] WARNING: ✗ Proxied tool missing: GITHUB_add_pull_request_review_comment
[11:11:37.469] WARNING: ✗ Proxied tool missing: GITHUB_create_branch
[11:11:37.469] WARNING: ✗ Proxied tool missing: GITHUB_create_issue
[11:11:37.469] WARNING: ✗ Proxied tool missing: GITHUB_create_or_update_file
[11:11:37.469] WARNING: ✗ Proxied tool missing: GITHUB_create_pull_request
[11:11:37.469] WARNING: ✗ Proxied tool missing: GITHUB_create_pull_request_review
[11:11:37.469] WARNING: ✗ Proxied tool missing: GITHUB_create_repository
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_fork_repository
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_code_scanning_alert
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_commit
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_file_contents
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_issue
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_issue_comments
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_me
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_pull_request
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_pull_request_comments
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_pull_request_files
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_pull_request_reviews
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_pull_request_status
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_secret_scanning_alert
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_get_tag
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_list_branches
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_list_code_scanning_alerts
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_list_commits
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_list_issues
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_list_pull_requests
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_list_secret_scanning_alerts
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_list_tags
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_merge_pull_request
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_push_files
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_search_code
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_search_issues
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_search_repositories
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_search_users
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_update_issue
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_update_pull_request
[11:11:37.470] WARNING: ✗ Proxied tool missing: GITHUB_update_pull_request_branch
[11:11:37.473] INFO: ✓ Integration test completed
```
