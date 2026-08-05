$hosts = @(
    "bufflehead-l","wigeon-l","smew-l","shoveler-l","eider-l","aylesbury-l",
    "barnacle-l","brent-l","cackling-l","canada-l","crested-l","gadwall-l",
    "goosander-l","gressingham-l","harlequin-l","mallard-l","mandarin-l",
    "pintail-l","pocher-l","ruddy-l","scaup-l","scoter-l","shelduck-l"
)

$remoteCmd = @'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=',' read -r pid mem; do
    pid=$(echo "$pid" | xargs)
    mem=$(echo "$mem" | xargs)
    owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
    cmd=$(ps -o cmd= -p "$pid" 2>/dev/null)
    echo "  PID $pid  owner=$owner  mem=$mem  cmd=$cmd"
done
'@
$remoteCmd = $remoteCmd -replace "`r`n", "`n"

foreach ($h in $hosts) {
    Write-Host "=== $h ===" -ForegroundColor Cyan
    $remoteCmd | & ssh -o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=accept-new $h "tr -d '\r' | bash"
    Write-Host "(exit code: $LASTEXITCODE)"
    Write-Host ""
    Start-Sleep -Milliseconds 400
}
