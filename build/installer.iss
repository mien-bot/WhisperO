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
CloseApplications=force
CloseApplicationsFilter=*.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupicon"; Description: "Start WhisperO when Windows starts"; GroupDescription: "Other:"

[Files]
Source: "..\dist\WhisperO\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "download_model.py"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: startupicon

; NOTE: No wholesale [InstallDelete] of {app}\* — that would erase any
; CUDA DLLs (cublas64_12.dll, cublasLt64_12.dll, cudart64_12.dll, ...)
; the user placed into the folder for GPU support. We rely on the
; `ignoreversion` flag in [Files] to overwrite updated files, and use
; BackupCudaDLLs / RestoreCudaDLLs in [Code] to preserve user-placed
; CUDA libs across reinstalls.

[Run]
; Reset device preference to GPU (fresh installs always default to GPU;
; overrides any previous device=cpu saved from a past CUDA-missing session).
Filename: "{app}\{#MyAppExeName}"; Parameters: "--reset-device-gpu"; Flags: waituntilterminated runhidden
; Download the whisper model after install (shown in a console window so user sees progress)
Filename: "{app}\{#MyAppExeName}"; Parameters: "--download-model"; StatusMsg: "Downloading speech recognition model (~3 GB, this may take a few minutes)..."; Flags: waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
var
  CudaBackupDir: String;

// CUDA DLL filenames users drop into the app folder for GPU support.
// These must survive reinstall/upgrade.
function GetCudaDllList(): TArrayOfString;
var
  L: TArrayOfString;
begin
  SetArrayLength(L, 6);
  L[0] := 'cublas64_12.dll';
  L[1] := 'cublasLt64_12.dll';
  L[2] := 'cudart64_12.dll';
  L[3] := 'cudnn_ops_infer64_8.dll';
  L[4] := 'cudnn_cnn_infer64_8.dll';
  L[5] := 'cudnn64_8.dll';
  Result := L;
end;

procedure BackupCudaDLLs();
var
  AppDir, SrcPath, DstPath: String;
  CudaDLLs: TArrayOfString;
  I: Integer;
begin
  AppDir := ExpandConstant('{app}');
  if not DirExists(AppDir) then Exit;

  CudaBackupDir := ExpandConstant('{tmp}') + '\whispero_cuda_backup';
  ForceDirectories(CudaBackupDir);

  CudaDLLs := GetCudaDllList();
  for I := 0 to GetArrayLength(CudaDLLs) - 1 do
  begin
    // Check both {app}\ and {app}\_internal\ for manually-placed DLLs
    SrcPath := AppDir + '\' + CudaDLLs[I];
    if FileExists(SrcPath) then
    begin
      DstPath := CudaBackupDir + '\' + CudaDLLs[I];
      FileCopy(SrcPath, DstPath, False);
    end;
    SrcPath := AppDir + '\_internal\' + CudaDLLs[I];
    if FileExists(SrcPath) then
    begin
      DstPath := CudaBackupDir + '\_internal_' + CudaDLLs[I];
      FileCopy(SrcPath, DstPath, False);
    end;
  end;
end;

procedure RestoreCudaDLLs();
var
  AppDir, SrcPath, DstPath: String;
  CudaDLLs: TArrayOfString;
  I: Integer;
begin
  if CudaBackupDir = '' then Exit;
  if not DirExists(CudaBackupDir) then Exit;

  AppDir := ExpandConstant('{app}');
  CudaDLLs := GetCudaDllList();
  for I := 0 to GetArrayLength(CudaDLLs) - 1 do
  begin
    // Only restore if the reinstall didn't already put one there
    // (don't clobber a DLL the new installer shipped).
    SrcPath := CudaBackupDir + '\' + CudaDLLs[I];
    DstPath := AppDir + '\' + CudaDLLs[I];
    if FileExists(SrcPath) and (not FileExists(DstPath)) then
      FileCopy(SrcPath, DstPath, False);

    SrcPath := CudaBackupDir + '\_internal_' + CudaDLLs[I];
    DstPath := AppDir + '\_internal\' + CudaDLLs[I];
    if FileExists(SrcPath) and (not FileExists(DstPath)) then
      FileCopy(SrcPath, DstPath, False);
  end;

  DelTree(CudaBackupDir, True, True, True);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
    BackupCudaDLLs()
  else if CurStep = ssPostInstall then
    RestoreCudaDLLs();
end;

function CheckInternet(): Boolean;
var
  WinHttpReq: Variant;
begin
  Result := False;
  try
    WinHttpReq := CreateOleObject('WinHttp.WinHttpRequest.5.1');
    WinHttpReq.Open('GET', 'https://huggingface.co', False);
    WinHttpReq.SetTimeouts(5000, 5000, 5000, 5000);
    WinHttpReq.Send('');
    Result := (WinHttpReq.Status = 200);
  except
    Result := False;
  end;
end;

function InitializeSetup(): Boolean;
var
  UninstallKey: String;
  UninstallString: String;
  ResultCode: Integer;
begin
  Result := True;

  // Check internet connectivity (required to download the speech model)
  if not CheckInternet() then
  begin
    if MsgBox('WhisperO requires an internet connection to download the speech recognition model (~3 GB) during installation.' + #13#10 + #13#10 +
              'No internet connection was detected. Do you want to continue anyway?' + #13#10 +
              '(The model will need to be downloaded later when you first launch the app.)',
              mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
      Exit;
    end;
  end;

  // Check for existing installation and silently uninstall it
  UninstallKey := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#SetupSetting("AppId")}_is1';
  if RegQueryStringValue(HKCU, UninstallKey, 'UninstallString', UninstallString) or
     RegQueryStringValue(HKLM, UninstallKey, 'UninstallString', UninstallString) then
  begin
    // Kill running WhisperO process before uninstalling
    Exec('taskkill', '/f /im {#MyAppExeName}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    // Run the old uninstaller silently
    Exec(RemoveQuotes(UninstallString), '/VERYSILENT /NORESTART /SUPPRESSMSGBOXES', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;
