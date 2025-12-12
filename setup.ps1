# MAIFDS - One-Command Setup Script (Windows PowerShell)
# This script sets up the entire project environment using uv

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  MAIFDS Project Setup" -ForegroundColor Cyan
Write-Host "  AI powered cyber protection for Ghana's MoMo ecosystem" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if uv is installed
try {
    $uvVersion = uv --version
    Write-Host "✅ uv found: $uvVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: uv is not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install uv first:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host ""

# Create virtual environment with uv (using Python 3.11 for MindSpore compatibility)
Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
Write-Host "   Note: MindSpore requires Python 3.9-3.11, using Python 3.11..." -ForegroundColor Gray
uv venv .venv --python 3.11
if ($LASTEXITCODE -ne 0) {
    uv venv .venv --python 3.10
    if ($LASTEXITCODE -ne 0) {
        uv venv .venv --python 3.9
        if ($LASTEXITCODE -ne 0) {
            uv venv .venv
        }
    }
}

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
Write-Host "   This may take a few minutes..." -ForegroundColor Gray
uv pip install -r requirements.txt

# Note: Skipping editable install - this is a monorepo with independent modules
# Each module can be run directly without package installation
Write-Host "✅ Setup complete! You can now run modules directly." -ForegroundColor Green

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  ✅ Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Or use uv to run commands directly:" -ForegroundColor Yellow
Write-Host "  uv run python <script.py>" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""

