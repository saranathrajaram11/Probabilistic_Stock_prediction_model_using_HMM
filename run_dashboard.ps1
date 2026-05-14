# Run from project folder:  .\run_dashboard.ps1
#
# - Uses `python -m pip` so you do not need `pip` on PATH.
# - Picks Python 3.12 / 3.11 / 3.10 via the Windows "py" launcher when available,
#   because hmmlearn often has no prebuilt wheel for very new Python (e.g. 3.14)
#   and would require Microsoft C++ Build Tools to compile from source.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Resolve-PythonExe {
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        foreach ($arg in @("-3.12", "-3.11", "-3.10")) {
            try {
                $exe = & py $arg -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $exe) {
                    Write-Host "Using Python from py launcher: py $arg -> $exe"
                    return $exe.Trim()
                }
            } catch {
                continue
            }
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        throw "No Python found. Install Python 3.12+ from https://www.python.org/downloads/windows/ and check 'Add python.exe to PATH'."
    }

    $verLine = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($verLine) {
        $mj, $mn = $verLine.Trim().Split(".")
        $major = [int]$mj
        $minor = [int]$mn
        if ($major -eq 3 -and $minor -ge 14) {
            Write-Warning @"
You are on Python $verLine. On Windows, hmmlearn may fail to install without MSVC Build Tools.
Fix: install Python 3.12 from python.org (and use 'py -3.12'), or install:
https://visualstudio.microsoft.com/visual-cpp-build-tools/
"@
        }
    }

    return "python"
}

$PythonExe = Resolve-PythonExe

Write-Host "Version check:"
& $PythonExe --version

Write-Host "Upgrading pip..."
& $PythonExe -m pip install --upgrade pip

# If pip fails with WinError 32 on a file under site-packages, another process
# (e.g. Streamlit/uvicorn or a second pip) has that file open — close it and retry.
Write-Host "Installing project dependencies..."
& $PythonExe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "pip install failed. Common fixes:" -ForegroundColor Yellow
    Write-Host "  1) Install Python 3.12 for prebuilt hmmlearn wheels:  py install 3.12" -ForegroundColor Yellow
    Write-Host "  2) Or install Microsoft C++ Build Tools if you must stay on Python 3.14+" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Starting Streamlit — open http://localhost:8501 in your browser"
Write-Host "(Use Ctrl+C in this window to stop the server.)"
Write-Host ""

& $PythonExe -m streamlit run dashboard.py --server.port 8501
