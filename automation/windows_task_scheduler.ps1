# ==============================================================================
# Скрипт регистрации автоматического расписания в Планировщике заданий Windows
# Сервис «Прозрачная вакансия» (КАНТКОД — Регион 39)
# ==============================================================================

param(
    [int]$IntervalMinutes = 60,
    [string]$TaskName = "KantKod_TransparentVacancy_Region39"
)

$ProjectDir = (Resolve-Path "$PSScriptRoot\..").Path
$PythonExe = (Get-Command python).Source

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "main.py live --max-vacancies 20 --page-limit 10" `
    -WorkingDirectory $ProjectDir

$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 365)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Description "Автоматическая загрузка и аудит прозрачности вакансий Калининградской области (регион 39) с портала Работа в России" `
    -Force

Write-Host "Задача '$TaskName' успешно зарегистрирована (интервал: каждые $IntervalMinutes мин.)."
