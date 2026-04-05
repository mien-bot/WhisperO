; WhisperO Windows Installer
; Requires Inno Setup 6+ — https://jrsoftware.org/isinfo.php
;
; Usage:
;   1. Build with PyInstaller first:  python build/build.py
;   2. Compile this script:           iscc build/installer.iss
;      (or open in Inno Setup GUI and click Compile)
;   Output: dist/WhisperO-Setup.exe

#define MyAppName      "WhisperO"
#define MyAppVersion   "0.1.0"
#define MyAppPublisher "Parker Cai"
#define MyAppExeName   "WhisperO.exe"

[Setup]
AppId={{B7E3F2A1-5C4D-4E6F-8A9B-1C2D3E4F5A6B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\dist
OutputBaseFilename=WhisperO-Setup
SetupIconFile=icons\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupicon"; Description: "Start WhisperO when Windows starts"; GroupDescription: "Other:"

[Files]
Source: "..\dist\WhisperO\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: startupicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
