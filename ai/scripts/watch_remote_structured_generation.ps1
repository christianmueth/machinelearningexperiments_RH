param(
    [string]$RemoteHost = "69.30.85.4",
    [int]$Port = 22166,
    [string]$RemoteUser = "root",
    [string]$KeyPath = "$HOME/.ssh/runpod_ed25519",
    [string]$RemoteDir = "/root",
    [Parameter(Mandatory = $true)]
    [string]$RemotePrefix,
    [Parameter(Mandatory = $true)]
    [string]$LocalDatasetPath,
    [string]$ArtifactPrefix = "",
    [int]$PollSeconds = 300,
    [string]$PythonExe = "",
    [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
if (-not $PythonExe) {
    $PythonExe = (Join-Path $RepoRoot ".venv\Scripts\python.exe")
}
if (-not $ArtifactPrefix) {
    $ArtifactPrefix = $RemotePrefix
}

$localDatasetFull = Join-Path $RepoRoot $LocalDatasetPath
$localDatasetDir = Split-Path -Parent $localDatasetFull
$localCandidateDir = Join-Path $RepoRoot "out\ai\candidate_generation\remote_sync"
New-Item -ItemType Directory -Force -Path $localDatasetDir | Out-Null
New-Item -ItemType Directory -Force -Path $localCandidateDir | Out-Null

$remoteDataset = "$RemoteDir/$RemotePrefix.jsonl"
$remoteConfig = "$RemoteDir/${RemotePrefix}_config.json"
$remoteSummary = "$RemoteDir/${RemotePrefix}_generation_summary.json"

$localConfig = Join-Path $localCandidateDir "${RemotePrefix}_config.json"
$localSummary = Join-Path $localCandidateDir "${RemotePrefix}_generation_summary.json"

function Invoke-Remote($RemoteCommand) {
    ssh -i $KeyPath -p $Port "$RemoteUser@$RemoteHost" $RemoteCommand
}

while ($true) {
    try {
        $status = Invoke-Remote "if [ -f '$remoteSummary' ]; then echo DONE; elif [ -f '$remoteDataset' ]; then wc -l '$remoteDataset'; else echo PENDING; fi"
        $trimmed = ($status | Out-String).Trim()
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "[$timestamp] remote status: $trimmed"
        if ($trimmed -eq "DONE") {
            scp -i $KeyPath -P $Port "$RemoteUser@${RemoteHost}:$remoteDataset" $localDatasetFull | Out-Host
            scp -i $KeyPath -P $Port "$RemoteUser@${RemoteHost}:$remoteConfig" $localConfig | Out-Host
            scp -i $KeyPath -P $Port "$RemoteUser@${RemoteHost}:$remoteSummary" $localSummary | Out-Host

            & $PythonExe (Join-Path $RepoRoot "ai\scripts\run_structured_controller_pipeline.py") --input_jsonl $LocalDatasetPath --artifact_prefix $ArtifactPrefix
            exit $LASTEXITCODE
        }
    }
    catch {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "[$timestamp] watcher retry after error: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $PollSeconds
}