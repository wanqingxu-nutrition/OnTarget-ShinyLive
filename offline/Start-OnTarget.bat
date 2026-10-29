@echo off
rem OnTarget - Diet Optimizer : double-click this file to run the calculator offline.
title OnTarget - Diet Optimizer
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\serve.ps1"
if errorlevel 1 pause
