# Setup script for ONNX Runtime GenAI Agent on Windows
# Run this script in PowerShell: .\setup.ps1

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "ONNX Runtime GenAI Agent - Setup" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found! Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv .venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip, wheel, and setuptools..." -ForegroundColor Yellow
python -m pip install --upgrade pip wheel setuptools --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "Upgraded successfully" -ForegroundColor Green
} else {
    Write-Host "Failed to upgrade" -ForegroundColor Red
}

# Install dependencies
Write-Host "`nInstalling dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Summary
Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Download an ONNX model (Phi-2, Phi-3, or Mistral)" -ForegroundColor White
Write-Host "     Example: huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Run the agent:" -ForegroundColor White
Write-Host "     python agent.py MODEL_PATH" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Or try interactive mode:" -ForegroundColor White
Write-Host "     python example_interactive.py MODEL_PATH" -ForegroundColor Gray
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor Cyan
Write-Host ""
