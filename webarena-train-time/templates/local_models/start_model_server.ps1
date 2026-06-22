param(
    [string]$Root = (Resolve-Path "$PSScriptRoot\..\..").Path,
    [string]$Python = "python",
    [ValidateSet("llama31_8b", "qwen3_14b", "qwen3_32b")]
    [string]$ModelProfile = "llama31_8b",
    [string]$ModelPath = "",
    [string[]]$Gpus = @("0"),
    [int]$Port = 11012,
    [string]$HostName = "127.0.0.1",
    [string]$DType = "",
    [ValidateSet("auto", "true", "false")]
    [string]$ChatTemplateEnableThinking = "auto",
    [switch]$TrustRemoteCode,
    [switch]$LoadIn4Bit,
    [switch]$LoadIn8Bit,
    [int]$MaxRepeatPrompt = 8
)

# Template launcher for the shared local Hugging Face model server.
# Example:
#   .\templates\local_models\start_model_server.ps1 -ModelProfile qwen3_14b -Gpus 0,1,2,3 -Port 11012

switch ($ModelProfile) {
    "llama31_8b" {
        if (-not $ModelPath) { $ModelPath = "meta-llama/Llama-3.1-8B-Instruct" }
        if (-not $DType) { $DType = "float16" }
        if ($ChatTemplateEnableThinking -eq "auto") { $ChatTemplateEnableThinking = "auto" }
    }
    "qwen3_14b" {
        if (-not $ModelPath) { $ModelPath = "Qwen/Qwen3-14B" }
        if (-not $DType) { $DType = "bfloat16" }
        if ($ChatTemplateEnableThinking -eq "auto") { $ChatTemplateEnableThinking = "false" }
    }
    "qwen3_32b" {
        if (-not $ModelPath) { $ModelPath = "Qwen/Qwen3-32B" }
        if (-not $DType) { $DType = "bfloat16" }
        if ($ChatTemplateEnableThinking -eq "auto") { $ChatTemplateEnableThinking = "false" }
    }
}

$server = Join-Path $Root "ahd-test-time\methods\eoh\original\eoh\src\eoh\llm_local_server\llama31_instruct_server.py"
$argsList = @(
    $server,
    "--path", $ModelPath,
    "--port", "$Port",
    "--host", $HostName,
    "--dtype", $DType,
    "--max-repeat-prompt", "$MaxRepeatPrompt",
    "--chat-template-enable-thinking", $ChatTemplateEnableThinking
)

if ($Gpus.Count -gt 0) {
    $argsList += "--d"
    $argsList += $Gpus
}
if ($TrustRemoteCode) { $argsList += "--trust-remote-code" }
if ($LoadIn4Bit) { $argsList += "--load-in-4bit" }
if ($LoadIn8Bit) { $argsList += "--quantization" }

Set-Location $Root
& $Python @argsList
