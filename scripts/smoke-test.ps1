[CmdletBinding()]
param(
    [string]$ApiKey = $env:RUNPOD_API_KEY,
    [string]$DeploymentFile = '.runpod-deployment.json',
    [int]$ReadyTimeoutMinutes = 45
)

$ErrorActionPreference = 'Stop'
if (-not $ApiKey) { throw 'Set RUNPOD_API_KEY or pass -ApiKey.' }
if (-not (Test-Path -LiteralPath $DeploymentFile)) { throw "Missing $DeploymentFile" }

$deployment = Get-Content -Raw $DeploymentFile | ConvertFrom-Json
$endpointId = $deployment.endpointId
$baseUrl = $deployment.requestUrls.base
$headers = @{ Authorization = "Bearer $ApiKey" }

# The first health request starts a scale-to-zero worker. Poll the worker-count endpoint,
# not the load balancer /ping route, while the image and model initialize.
$healthUrl = "https://api.runpod.ai/v2/$endpointId/health"
$deadline = (Get-Date).AddMinutes($ReadyTimeoutMinutes)
do {
    $health = Invoke-RestMethod -Uri $healthUrl -Headers $headers
    Write-Host ("Worker state: " + ($health.workers | ConvertTo-Json -Compress))
    if ($health.workers.ready -ge 1) { break }
    Start-Sleep -Seconds 10
} while ((Get-Date) -lt $deadline)
if ($health.workers.ready -lt 1) { throw "Endpoint did not become ready within $ReadyTimeoutMinutes minutes." }

$models = Invoke-RestMethod -Uri "$baseUrl/v1/models" -Headers $headers

$textBody = @{
    model = 'qwen3.8-27b-huihui-abliterated'
    messages = @(@{ role = 'user'; content = 'Think carefully: what is 17 multiplied by 19? Give only the result.' })
    reasoning_effort = 'xhigh'
    max_tokens = 128
    stream = $false
} | ConvertTo-Json -Depth 10
$text = Invoke-RestMethod -Method Post -Uri "$baseUrl/v1/chat/completions" `
    -Headers $headers -ContentType 'application/json' -Body $textBody -TimeoutSec 330

$visionBody = @{
    model = 'qwen3.8-27b-huihui-abliterated'
    messages = @(@{
        role = 'user'
        content = @(
            @{ type = 'text'; text = 'What animals are visible? Answer in one short sentence.' },
            @{ type = 'image_url'; image_url = @{ url = 'https://images.cocodataset.org/val2017/000000039769.jpg' } }
        )
    })
    reasoning_effort = 'low'
    max_tokens = 128
    stream = $false
} | ConvertTo-Json -Depth 20
$vision = Invoke-RestMethod -Method Post -Uri "$baseUrl/v1/chat/completions" `
    -Headers $headers -ContentType 'application/json' -Body $visionBody -TimeoutSec 330

[ordered]@{
    health = $health
    models = $models
    text = $text
    vision = $vision
} | ConvertTo-Json -Depth 30
