$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Project environment not found at $python"
}
& $python -m streamlit run (Join-Path $PSScriptRoot "starter\app.py")
