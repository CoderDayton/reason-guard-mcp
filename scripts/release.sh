#!/usr/bin/env bash
# Release script for MatrixMind MCP
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 0.2.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Validate arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <version> [--dry-run]"
    echo "Example: $0 0.2.0"
    echo "         $0 0.2.0 --dry-run"
    exit 1
fi

VERSION="$1"
DRY_RUN="${2:-}"

# Validate version format
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$'; then
    error "Invalid version format: $VERSION (expected X.Y.Z or X.Y.Z-suffix)"
fi

# Check we're in the repo root
if [ ! -f "pyproject.toml" ]; then
    error "Must be run from repository root (pyproject.toml not found)"
fi

# Check for clean working directory
if [ -n "$(git status --porcelain)" ]; then
    error "Working directory is not clean. Commit or stash changes first."
fi

# Check we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    warn "Not on main branch (currently on $CURRENT_BRANCH)"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if tag already exists
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    error "Tag v$VERSION already exists"
fi

info "Preparing release v$VERSION"

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep -oP 'version = "\K[^"]+' pyproject.toml | head -1)
info "Current version: $CURRENT_VERSION"
info "New version: $VERSION"

if [ "$DRY_RUN" = "--dry-run" ]; then
    warn "DRY RUN MODE - No changes will be made"
    echo ""
    echo "Would perform the following:"
    echo "  1. Update pyproject.toml version from $CURRENT_VERSION to $VERSION"
    echo "  2. Commit with message: chore(release): bump version to $VERSION"
    echo "  3. Create signed tag: v$VERSION"
    echo "  4. Push commit and tag to origin"
    echo ""
    echo "This would trigger the release workflow to:"
    echo "  - Run tests"
    echo "  - Build package"
    echo "  - Sign artifacts with cosign"
    echo "  - Publish to PyPI"
    echo "  - Build and push Docker image to GHCR"
    echo "  - Create GitHub release"
    exit 0
fi

# Confirm release
echo ""
echo "This will:"
echo "  1. Update pyproject.toml version to $VERSION"
echo "  2. Commit and create signed tag v$VERSION"
echo "  3. Push to origin (triggers release workflow)"
echo ""
read -p "Proceed with release v$VERSION? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    info "Release cancelled"
    exit 0
fi

# Update version in pyproject.toml
info "Updating pyproject.toml..."
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# Verify the change
NEW_VERSION=$(grep -oP 'version = "\K[^"]+' pyproject.toml | head -1)
if [ "$NEW_VERSION" != "$VERSION" ]; then
    error "Failed to update version in pyproject.toml"
fi
success "Updated pyproject.toml to version $VERSION"

# Run tests before committing
info "Running tests..."
if command -v uv &> /dev/null; then
    uv run pytest tests/ -q --tb=no -x || error "Tests failed. Reverting changes."
else
    python -m pytest tests/ -q --tb=no -x || error "Tests failed. Reverting changes."
fi
success "Tests passed"

# Commit the version bump
info "Committing version bump..."
git add pyproject.toml
git commit -m "chore(release): bump version to $VERSION"
success "Committed version bump"

# Create signed tag
info "Creating signed tag v$VERSION..."
if git config --get user.signingkey &> /dev/null; then
    # GPG signing available
    git tag -s "v$VERSION" -m "Release v$VERSION"
    success "Created signed tag v$VERSION (GPG)"
else
    # Fall back to annotated tag (cosign will sign in CI)
    git tag -a "v$VERSION" -m "Release v$VERSION"
    success "Created annotated tag v$VERSION (will be signed by cosign in CI)"
fi

# Push commit and tag
info "Pushing to origin..."
git push origin main
git push origin "v$VERSION"
success "Pushed commit and tag to origin"

echo ""
success "Release v$VERSION initiated!"
echo ""
echo "The release workflow is now running. Monitor progress at:"
echo "  https://github.com/CoderDayton/matrixmind-mcp/actions"
echo ""
echo "Once complete, the release will be available at:"
echo "  - PyPI: https://pypi.org/project/matrixmind-mcp/$VERSION/"
echo "  - GitHub: https://github.com/CoderDayton/matrixmind-mcp/releases/tag/v$VERSION"
echo "  - Docker: ghcr.io/coderdayton/matrixmind-mcp:$VERSION"
