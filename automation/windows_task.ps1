# Регистрирует в Планировщике заданий Windows ежедневный запуск рабочего режима.
# Запуск:   powershell -ExecutionPolicy Bypass -File automation\windows_task.ps1 [-At 06:00]
# Удаление: Unregister-ScheduledTask -TaskName "ProzrachnayaVakansiya" -Confirm:$false
param([string]$At = "06:00")

$root = Split-Path -Parent $PSScriptRoot
$uv = (Get-Command uv -ErrorAction Stop).Source
$action = New-ScheduledTaskAction -Execute $uv -Argument "run python -m pv run" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName "ProzrachnayaVakansiya" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Prozrachnaya vakansiya: daily vacancy report" -Force | Select-Object TaskName, State
