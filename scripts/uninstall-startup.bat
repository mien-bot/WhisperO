@echo off
echo Removing WhisperO from Windows Startup...
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\WhisperO.vbs"
if %errorlevel% == 0 (
    echo Done! WhisperO will no longer start automatically.
) else (
    echo Nothing to remove.
)
pause
