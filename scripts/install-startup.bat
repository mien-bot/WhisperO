@echo off
echo Installing WhisperO to Windows Startup...
copy "%~dp0WhisperO.vbs" "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\WhisperO.vbs"
if %errorlevel% == 0 (
    echo Done! WhisperO will start automatically on login.
) else (
    echo Failed to install. Try running as Administrator.
)
pause
