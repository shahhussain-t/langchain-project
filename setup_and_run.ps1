# AI Assistant Setup and Run Script
# This script ensures virtual environment is active and checks Python version

Write-Host "🤖 AI Assistant Setup Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check Python version
Write-Host "`n📋 Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Current Python version: $pythonVersion" -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "`n🔧 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Virtual environment created successfully!" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n🚀 Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Check if we're in virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "✅ Virtual environment activated: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}

# Check disk space
Write-Host "`n💾 Checking available disk space..." -ForegroundColor Yellow
try {
    $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='E:'" -ErrorAction SilentlyContinue
    if ($disk) {
        $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2)
        $totalSpaceGB = [math]::Round($disk.Size / 1GB, 2)
        Write-Host "E: Drive - Free: ${freeSpaceGB}GB / Total: ${totalSpaceGB}GB" -ForegroundColor Green
        
        if ($freeSpaceGB -lt 2) {
            Write-Host "⚠️  Warning: Low disk space! Consider cleaning up files." -ForegroundColor Red
        }
    }
} catch {
    Write-Host "Could not check disk space" -ForegroundColor Yellow
}

# Install/Update dependencies if needed
Write-Host "`n📦 Checking backend dependencies..." -ForegroundColor Yellow
Set-Location backend

# Check if requirements are installed
try {
    $fastapi = pip show fastapi 2>$null
    if (-not $fastapi) {
        Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
        pip install fastapi uvicorn[standard] python-dotenv pdfplumber python-docx openai pinecone langchain langchain-openai langchain-community
        Write-Host "✅ Dependencies installed!" -ForegroundColor Green
    } else {
        Write-Host "✅ Dependencies already installed" -ForegroundColor Green
    }
} catch {
    Write-Host "Error checking dependencies" -ForegroundColor Red
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "`n⚠️  .env file not found. Please create one with your API keys:" -ForegroundColor Yellow
    Write-Host "OPENAI_API_KEY=your_openai_api_key_here" -ForegroundColor White
    Write-Host "PINECONE_API_KEY=your_pinecone_api_key_here" -ForegroundColor White
    Write-Host "`n📝 A template .env file has been created for you." -ForegroundColor Green
} else {
    Write-Host "`n✅ .env file found" -ForegroundColor Green
}

# Setup frontend if Node.js is available
Set-Location ../frontend
Write-Host "`n🌐 Setting up frontend..." -ForegroundColor Yellow

try {
    $nodeVersion = node --version 2>$null
    if ($nodeVersion) {
        Write-Host "Node.js version: $nodeVersion" -ForegroundColor Green
        
        # Check if node_modules exists
        if (-not (Test-Path "node_modules")) {
            Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
            npm install
            Write-Host "✅ Frontend dependencies installed!" -ForegroundColor Green
        } else {
            Write-Host "✅ Frontend dependencies already installed" -ForegroundColor Green
        }
    } else {
        Write-Host "⚠️  Node.js not found. Please install Node.js to run the frontend." -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Node.js not found. Please install Node.js to run the frontend." -ForegroundColor Yellow
}

Set-Location ..

Write-Host "`n🎯 Setup complete! You can now:" -ForegroundColor Cyan
Write-Host "1. Add your API keys to backend\.env" -ForegroundColor White
Write-Host "2. Backend: cd backend && python main.py" -ForegroundColor White
Write-Host "3. Frontend: cd frontend && npm start" -ForegroundColor White
Write-Host "`n💡 Quick commands:" -ForegroundColor Cyan
Write-Host "- Run this script anytime: .\setup_and_run.ps1" -ForegroundColor White
Write-Host "- Quick setup: double-click quick_setup.bat" -ForegroundColor White