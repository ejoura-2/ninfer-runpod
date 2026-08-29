[CmdletBinding()]
param(
    [string]$ApiKey = $env:RUNPOD_API_KEY,
    [string]$Image = 'ghcr.io/ejoura-2/ninfer-runpod:latest',
    [string]$ModelReference = 'https://huggingface.co/lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4:181446902fc777c479749e98cf2abf2250263a8d'
)

$ErrorActionPreference = 'Stop'
if (-not $ApiKey) {
    throw 'Set RUNPOD_API_KEY or pass -ApiKey. The key is never written to disk.'
}

$headers = @{ Authorization = "Bearer $ApiKey" }
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$templateName = "ninfer-qwen38-queue-$timestamp"
$templateId = $null
$endpointId = $null

$envConfig = @{
    MODEL_REPO_ID = 'lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4'
    MODEL_FILENAME = 'qwen3_8_27b_nvfp4.ninfer'
    MODEL_ID = 'qwen3.8-27b-huihui-abliterated'
    MAX_CONTEXT = '204800'
    KV_CAPACITY = '204800'
    KV_DTYPE = 'int8'
    MAX_CONCURRENCY = '1'
    PREFILL_CHUNK = '4096'
    DEFAULT_MAX_TOKENS = '32768'
    DRAFT_TOKENS = '3'
    PORT = '8080'
    RUNPOD_INIT_TIMEOUT = '1200'
    REQUEST_TIMEOUT = '3600'
}

try {
    $templateBody = @{
        category = 'NVIDIA'
        containerDiskInGb = 80
        dockerEntrypoint = @()
        dockerStartCmd = @()
        env = $envConfig
        imageName = $Image
        isPublic = $false
        isServerless = $true
        name = $templateName
        ports = @()
        readme = 'NInfer Qwen3.8-27B Huihui Abliterated NVFP4 queue worker with OpenAI passthrough.'
        volumeInGb = 0
        volumeMountPath = '/runpod-volume'
    } | ConvertTo-Json -Depth 10

    $template = Invoke-RestMethod -Method Post `
        -Uri 'https://rest.runpod.io/v1/templates' `
        -Headers $headers -ContentType 'application/json' -Body $templateBody
    $templateId = $template.id
    Write-Host "Created serverless template: $templateId"

    $graphQuery = @'
mutation SaveEndpoint($input: EndpointInput!) {
  saveEndpoint(input: $input) {
    id name type templateId gpuIds gpuCount workersMin workersMax idleTimeout
    scalerType scalerValue executionTimeoutMs flashBootType modelReferences
  }
}
'@
    $variables = @{
        input = @{
            name = 'ninfer-qwen38-27b-abliterated'
            templateId = $templateId
            type = 'QB'
            gpuIds = 'ADA_32_PRO'
            gpuCount = 1
            workersMin = 0
            workersMax = 1
            idleTimeout = 300
            scalerType = 'QUEUE_DELAY'
            scalerValue = 4
            executionTimeoutMs = 3600000
            flashBootType = 'FLASHBOOT'
            modelReferences = @($ModelReference)
        }
    }
    $graphBody = @{ query = $graphQuery; variables = $variables } | ConvertTo-Json -Depth 20
    $graph = Invoke-RestMethod -Method Post -Uri 'https://api.runpod.io/graphql' `
        -Headers ($headers + @{ 'User-Agent' = 'Mozilla/5.0' }) `
        -ContentType 'application/json' -Body $graphBody
    if ($graph.errors) {
        throw ($graph.errors | ConvertTo-Json -Depth 10 -Compress)
    }
    $endpointId = $graph.data.saveEndpoint.id
    Write-Host "Created queue endpoint: $endpointId"

    $gpuPatch = @{
        gpu = @{
            pools = @('ADA_32_PRO')
            excludedTypes = @()
            count = 1
            allowedCudaVersions = @('13.0')
            minCudaVersion = ''
        }
    } | ConvertTo-Json -Depth 10
    $endpoint = Invoke-RestMethod -Method Patch `
        -Uri "https://api.runpod.io/v2/serverless/$endpointId" `
        -Headers $headers -ContentType 'application/json' -Body $gpuPatch

    $deployment = [ordered]@{
        endpointId = $endpoint.id
        endpointName = $endpoint.name
        endpointType = $endpoint.type
        templateId = $templateId
        image = $Image
        modelReference = $ModelReference
        gpu = $endpoint.gpu
        workers = $endpoint.workers
        scaling = $endpoint.scaling
        requestUrls = $endpoint.requestUrls
        createdAt = (Get-Date).ToUniversalTime().ToString('o')
    }
    $deployment | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 '.runpod-deployment.json'
    $deployment | ConvertTo-Json -Depth 20
}
catch {
    Write-Error $_
    if ($templateId -and -not $endpointId) {
        Write-Warning "Endpoint creation failed; deleting unused template $templateId."
        Invoke-RestMethod -Method Delete -Uri "https://rest.runpod.io/v1/templates/$templateId" -Headers $headers | Out-Null
    }
    throw
}
