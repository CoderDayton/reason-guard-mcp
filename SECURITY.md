# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting feature
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 7 days
  - Medium: 30 days
  - Low: 90 days

### Scope

The following are in scope for security reports:

- Authentication/authorization bypasses
- Injection vulnerabilities (prompt injection, code injection)
- Sensitive data exposure (API keys, credentials)
- Denial of service vulnerabilities
- Supply chain vulnerabilities in dependencies

### Out of Scope

- Issues in dependencies (report to upstream maintainers)
- Social engineering attacks
- Physical security issues
- Issues requiring unlikely user interaction

## Security Best Practices for Users

### API Key Management

```bash
# ✅ DO: Use environment variables
export OPENAI_API_KEY=sk-...

# ✅ DO: Use .env files (excluded from git)
echo "OPENAI_API_KEY=sk-..." > .env

# ❌ DON'T: Hardcode in source files
# ❌ DON'T: Commit .env files to git
```

### Running in Production

1. **Use read-only filesystem** where possible
2. **Run as non-root user** (Dockerfile already configured)
3. **Limit network access** to required endpoints only
4. **Enable container scanning** in your CI/CD pipeline
5. **Verify image signatures** before deployment:

```bash
cosign verify \
  --certificate-identity-regexp="https://github.com/OWNER/enhanced-cot-mcp.*" \
  --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
  ghcr.io/OWNER/enhanced-cot-mcp:latest
```

### Dependency Management

- Pin dependencies via `uv.lock`
- Regularly update dependencies: `uv lock --upgrade`
- Review dependency changes before merging
- Use `uv audit` or similar tools to check for known vulnerabilities

## Security Features

### Supply Chain Security

- **Signed container images**: All releases are signed with Sigstore cosign
- **Lockfile pinning**: `uv.lock` ensures reproducible builds
- **SBOM generation**: Available via container image labels

### Code Security

- **Pre-commit hooks**: Detect secrets before commit
- **Bandit scanning**: Static analysis for Python security issues
- **Type checking**: mypy catches type-related bugs

### Secret Protection

This repository uses:
- `.gitignore` to exclude `.env` files
- Pre-commit hooks with `detect-private-key`
- GitHub secret scanning (if enabled on repository)

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who report valid vulnerabilities (with permission).
