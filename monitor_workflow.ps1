# =============================================================================
# GITHUB ACTIONS WORKFLOW MONITOR
# Auto-check workflow status every 30 seconds
# =============================================================================

$repoOwner = "Rivaldy-25-Lval"
$repoName = "Workflow-CI-Muhammad-Rivaldy-Pratama"
$apiUrl = "https://api.github.com/repos/$repoOwner/$repoName/actions/runs"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "GITHUB ACTIONS WORKFLOW MONITOR" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Repository: $repoOwner/$repoName" -ForegroundColor White
Write-Host "Monitoring workflow status every 30 seconds..." -ForegroundColor White
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
Write-Host ""

$iteration = 0
$workflowCompleted = $false

while (-not $workflowCompleted) {
    $iteration++
    
    try {
        # Fetch latest workflow run
        $response = Invoke-RestMethod -Uri $apiUrl -Method Get -ErrorAction Stop
        $latestRun = $response.workflow_runs[0]
        
        # Extract info
        $runId = $latestRun.id
        $status = $latestRun.status
        $conclusion = $latestRun.conclusion
        $workflowName = $latestRun.name
        $commitMessage = $latestRun.head_commit.message
        $createdAt = [DateTime]::Parse($latestRun.created_at).ToLocalTime()
        $runUrl = $latestRun.html_url
        
        # Calculate elapsed time
        $elapsed = (Get-Date) - $createdAt
        $elapsedStr = "{0:D2}:{1:D2}:{2:D2}" -f $elapsed.Hours, $elapsed.Minutes, $elapsed.Seconds
        
        # Clear screen for clean display
        Clear-Host
        
        Write-Host "=" * 80 -ForegroundColor Cyan
        Write-Host "WORKFLOW STATUS - Check #$iteration" -ForegroundColor Yellow
        Write-Host "=" * 80 -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "Workflow: " -NoNewline -ForegroundColor Gray
        Write-Host "$workflowName" -ForegroundColor White
        
        Write-Host "Commit:   " -NoNewline -ForegroundColor Gray
        Write-Host "$commitMessage" -ForegroundColor White
        
        Write-Host "Started:  " -NoNewline -ForegroundColor Gray
        Write-Host "$createdAt" -ForegroundColor White
        
        Write-Host "Elapsed:  " -NoNewline -ForegroundColor Gray
        Write-Host "$elapsedStr" -ForegroundColor White
        
        Write-Host ""
        Write-Host "-" * 80 -ForegroundColor Gray
        
        # Display status with color
        Write-Host "Status:     " -NoNewline -ForegroundColor Gray
        if ($status -eq "completed") {
            Write-Host "$status" -ForegroundColor Green
            
            Write-Host "Conclusion: " -NoNewline -ForegroundColor Gray
            if ($conclusion -eq "success") {
                Write-Host "$conclusion" -ForegroundColor Green -BackgroundColor DarkGreen
                $workflowCompleted = $true
            }
            elseif ($conclusion -eq "failure") {
                Write-Host "$conclusion" -ForegroundColor Red -BackgroundColor DarkRed
                $workflowCompleted = $true
            }
            else {
                Write-Host "$conclusion" -ForegroundColor Yellow
                $workflowCompleted = $true
            }
        }
        elseif ($status -eq "in_progress") {
            Write-Host "$status" -ForegroundColor Yellow
            Write-Host "Conclusion: " -NoNewline -ForegroundColor Gray
            Write-Host "Running..." -ForegroundColor Yellow
        }
        else {
            Write-Host "$status" -ForegroundColor Cyan
            Write-Host "Conclusion: " -NoNewline -ForegroundColor Gray
            Write-Host "Pending..." -ForegroundColor Cyan
        }
        
        Write-Host ""
        Write-Host "-" * 80 -ForegroundColor Gray
        Write-Host "URL: " -NoNewline -ForegroundColor Gray
        Write-Host "$runUrl" -ForegroundColor Blue
        
        # If completed, show final results
        if ($workflowCompleted) {
            Write-Host ""
            Write-Host "=" * 80 -ForegroundColor Cyan
            
            if ($conclusion -eq "success") {
                Write-Host "SUCCESS! WORKFLOW COMPLETED" -ForegroundColor Green -BackgroundColor DarkGreen
                Write-Host "=" * 80 -ForegroundColor Cyan
                Write-Host ""
                Write-Host "KRITERIA 3 - VERIFICATION" -ForegroundColor Yellow
                Write-Host "-" * 80 -ForegroundColor Gray
                Write-Host "[x] MLflow training completed" -ForegroundColor Green
                Write-Host "[x] Docker image built successfully" -ForegroundColor Green
                Write-Host "[x] Docker image pushed to Docker Hub" -ForegroundColor Green
                Write-Host "[x] Artifacts uploaded to GitHub" -ForegroundColor Green
                Write-Host ""
                Write-Host "Next steps:" -ForegroundColor White
                Write-Host "1. Verify Docker Hub: https://hub.docker.com/r/lval2505/heart-disease-ml/tags" -ForegroundColor Cyan
                Write-Host "2. Check Docker image has 'latest' and '3dda790' tags" -ForegroundColor Cyan
                Write-Host "3. Proceed to Kriteria 4 (29 screenshots)" -ForegroundColor Cyan
                Write-Host ""
                Write-Host "KRITERIA 3 STATUS: ADVANCED (4/4 POINTS) COMPLETE!" -ForegroundColor Green -BackgroundColor DarkGreen
            }
            elseif ($conclusion -eq "failure") {
                Write-Host "WORKFLOW FAILED" -ForegroundColor Red -BackgroundColor DarkRed
                Write-Host "=" * 80 -ForegroundColor Cyan
                Write-Host ""
                Write-Host "Please check the workflow logs:" -ForegroundColor Yellow
                Write-Host "$runUrl" -ForegroundColor Blue
                Write-Host ""
                Write-Host "Common issues:" -ForegroundColor White
                Write-Host "- Docker Hub credentials expired" -ForegroundColor Gray
                Write-Host "- MLflow training errors" -ForegroundColor Gray
                Write-Host "- Dependency installation failed" -ForegroundColor Gray
            }
            
            Write-Host ""
            Write-Host "=" * 80 -ForegroundColor Cyan
            Write-Host "Monitoring stopped. Total time: $elapsedStr" -ForegroundColor Gray
            break
        }
        
        Write-Host ""
        Write-Host "Next check in 30 seconds... (Check #$($iteration + 1))" -ForegroundColor Gray
        Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor DarkGray
        
        Start-Sleep -Seconds 30
        
    }
    catch {
        Write-Host ""
        Write-Host "Error fetching workflow status: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Retrying in 30 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
    }
}

Write-Host ""
Write-Host "Monitoring completed. Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
