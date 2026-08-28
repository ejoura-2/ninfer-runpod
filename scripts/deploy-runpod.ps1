[CmdletBinding()]
param(
    [string]$ApiKey = $env:RUNPOD_API_KEY,
    [string]$Image = 'ghcr.io/ejoura-2/ninfer-runpod:latest',
    [string]$ModelReference = 'https://huggingface.co/lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4:main'
)

$ErrorActionPreference = 'Stop'
if (-not $ApiKey) {
    throw 'Set RUNPOD_API_KEY or pass -ApiKey. The key is never written to disk.'
}

$headers = @{ Authorization = "Bearer $ApiKey" }
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$templateName = "ninfer-qwen38-lb-$timestamp"
$templateId = $null
$endpointId = $null

$envConfig = @{
    MODEL_REPO_ID = 'lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4'
    MODEL_FILENAME = 'qwen3_8_27b_nvfp4.ninfer'
    MODEL_ID = 'qwen3.8-27b-huihui-abliterated'
    MAX_CONTEXT = '262144'
    KV_CAPACITY = '262144'
    KV_DTYPE = 'fp8'
    MAX_CONCURRENCY = '1'
    PREFILL_CHUNK = '4096'
    DEFAULT_MAX_TOKENS = '32768'
    DRAFT_TOKENS = '3'
    PORT = '8080'
    PORT_HEALTH = '8081'
    RUNPOD_INIT_TIMEOUT = '800'
    HEALTH_CHECK_PATH = '/ping'
}

try {
    $templateBody = @{
        category = 'NVIDIA'
        containerDiskInGb = 10
        dockerEntrypoint = @()
        dockerStartCmd = @()
        env = $envConfig
        imageName = $Image
        isPublic = $false
        isServerless = $true
        name = $templateName
        ports = @('8080/http', '8081/http')
        readme = 'NInfer Qwen3.8-27B Huihui Abliterated NVFP4 load-balancing worker.'
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
            type = 'LB'
            gpuIds = 'BLACKWELL_96'
            gpuCount = 1
            workersMin = 0
            workersMax = 1
            idleTimeout = 60
            scalerType = 'REQUEST_COUNT'
            scalerValue = 1
            executionTimeoutMs = 330000
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
    Write-Host "Created load-balancing endpoint: $endpointId"

    $gpuPatch = @{
        gpu = @{
            pools = @('BLACKWELL_96')
            excludedTypes = @(
                'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition',
                'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'
            )
            count = 1
            allowedCudaVersions = @('13.2')
            # GraphQL endpoint creation defaults this to 12.0. REST v2 requires the
            # inherited floor to be cleared in the same request as an exact allowlist.
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
