# Branch Protection Configuration

Configure these settings in GitHub: Settings → Branches → Add branch protection rule

## Rule: `main`

### Protect matching branches

- [x] **Require a pull request before merging**
  - [ ] Require approvals (optional, set to 1 for teams)
  - [x] Dismiss stale pull request approvals when new commits are pushed

- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - Required status checks:
    - `Lint`
    - `Type Check`
    - `Test`
    - `Build`
    - `Changelog Validation` (for PRs)

- [x] **Require conversation resolution before merging**

- [x] **Do not allow bypassing the above settings**

### Rules applied to everyone including administrators

- [ ] Allow force pushes (keep disabled)
- [ ] Allow deletions (keep disabled)

## Automated Setup (GitHub CLI)

```bash
gh api repos/{owner}/{repo}/branches/main/protection -X PUT \
  -H "Accept: application/vnd.github+json" \
  -f required_status_checks='{"strict":true,"contexts":["Lint","Type Check","Test","Build"]}' \
  -f enforce_admins=true \
  -f required_pull_request_reviews='{"dismiss_stale_reviews":true}' \
  -f restrictions=null
```
