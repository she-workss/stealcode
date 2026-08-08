#ifndef Version
  #define Version "0.0.0"
#endif
#ifndef Channel
  #define Channel "stable"
#endif
#ifndef ResourcesDir
  ; Falls back to "this file's own directory" so a developer can compile
  ; directly with ISCC for a quick local check, without running the full
  ; bundling script first.
  #define ResourcesDir SourcePath
#endif
#ifndef OutputDir
  ; Falls back to "Output" so a developer can compile directly with ISCC
  #define OutputDir "Output"
#endif

#define NumericVersion Version

#if Copy(NumericVersion, 1, 1) == "v"
  #define NumericVersion Copy(NumericVersion, 2, Len(NumericVersion) - 1)
#endif
#if Pos('-', NumericVersion) > 0
  #define NumericVersion Copy(NumericVersion, 1, Pos('-', NumericVersion) - 1)
#endif

#if Pos('-', Version) > 0
  #define NumericVersion Copy(Version, 1, Pos('-', Version) - 1)
#elif Pos('+', Version) > 0
  #define NumericVersion Copy(Version, 1, Pos('+', Version) - 1)
#endif
#if Pos('+', NumericVersion) > 0
  #define NumericVersion Copy(NumericVersion, 1, Pos('+', NumericVersion) - 1)
#endif

#if Channel == "nightly"
  #define AppId "{{E44A5546-BEA9-43F6-AD18-39A7452712B5}"
  #define MyAppName "StealCode Nightly"
  #define AppUserId "he-thinks.StealCode.Nightly"
  #define DirSuffix " Nightly"
#else
  #define AppId "{{0A5E2833-6A6A-473E-AE3B-2A887A6ECDD8}"
  #define MyAppName "StealCode"
  #define AppUserId "he-thinks.StealCode"
  #define DirSuffix ""
#endif

#define MyAppPublisher "he-thinks"
#define MyAppURL "https://github.com/she-workss/stealcode"
#define MyAppExeName "stealcode.exe"

[Setup]
AppId={#AppId}
AppName={#MyAppName}
AppVersion={#Version}
VersionInfoVersion={#NumericVersion}
VersionInfoProductVersion={#NumericVersion}
VersionInfoProductTextVersion={#Version}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir={#OutputDir}
LicenseFile={#ResourcesDir}LICENSE
InfoBeforeFile={#ResourcesDir}README.md
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
OutputBaseFilename=StealCode-{#MyAppArch}
SetupIconFile={#ResourcesDir}icon.ico
SolidCompression=yes
WizardStyle=modern dynamic
ChangesEnvironment=true
ChangesAssociations=true
DisableReadyPage=yes
CloseApplications=force

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl,{#ResourcesDir}messages\en.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl,{#ResourcesDir}messages\ru.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "{cm:AddToPath}"; GroupDescription: "{cm:Other}"
Name: "addcontextmenu"; Description: "{cm:AddContextMenu}"; GroupDescription: "{cm:Other}"

[Dirs]
Name: "{app}"; AfterInstall: DisableAppDirInheritance
Name: "{app}\tools"

[Files]
Source: "{#ResourcesDir}{#MyAppExeName}"; DestDir: "{code:GetInstallDir}"; Flags: ignoreversion
Source: "{#ResourcesDir}auto_update_helper.exe"; DestDir: "{app}\tools"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; AppUserModelID: "{#AppUserId}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; AppUserModelID: "{#AppUserId}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall; Check: WizardNotSilent

[Registry]
; PATH
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{code:AddToPath|{app}}"; Tasks: addtopath; Check: NeedsAddToPath(ExpandConstant('{app}'))

; Context menu: folder as an object
Root: HKCU; Subkey: "Software\Classes\Directory\shell\StealCode{#DirSuffix}"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:ContextMenuGroup}"; Flags: uninsdeletekey; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\StealCode{#DirSuffix}"; ValueType: string; ValueName: "SubCommands"; ValueData: ""; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\StealCode{#DirSuffix}"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"""; Tasks: addcontextmenu

Root: HKCU; Subkey: "Software\Classes\Directory\shell\StealCode{#DirSuffix}\shell\01console"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenuConsole}"; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\StealCode{#DirSuffix}\shell\01console"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"""; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\StealCode{#DirSuffix}\shell\01console\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: addcontextmenu

Root: HKCU; Subkey: "Software\Classes\Directory\shell\StealCode{#DirSuffix}\shell\02desktop"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenuDesktop}"; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\StealCode{#DirSuffix}\shell\02desktop"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"""; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\StealCode{#DirSuffix}\shell\02desktop\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" desktop ""%1"""; Tasks: addcontextmenu

; Context menu: folder background
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\StealCode{#DirSuffix}"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:ContextMenuGroup}"; Flags: uninsdeletekey; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\StealCode{#DirSuffix}"; ValueType: string; ValueName: "SubCommands"; ValueData: ""; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\StealCode{#DirSuffix}"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"""; Tasks: addcontextmenu

Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\StealCode{#DirSuffix}\shell\01console"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenuConsole}"; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\StealCode{#DirSuffix}\shell\01console"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"""; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\StealCode{#DirSuffix}\shell\01console\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%V"""; Tasks: addcontextmenu

Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\StealCode{#DirSuffix}\shell\02desktop"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenuDesktop}"; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\StealCode{#DirSuffix}\shell\02desktop"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"""; Tasks: addcontextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\StealCode{#DirSuffix}\shell\02desktop\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" desktop ""%V"""; Tasks: addcontextmenu

[Code]
var
  RemoveSettingsCheckBox, RemoveLogsCheckBox: TNewCheckBox;
  RemoveSettingsChecked, RemoveLogsChecked: Boolean;

function WizardNotSilent(): Boolean;
begin
  Result := not WizardSilent();
end;

procedure InitializeUninstallProgressForm();
var
  UninstallPage: TNewNotebookPage;
  UninstallButton: TNewButton;
  OriginalPageNameLabel, OriginalPageDescriptionLabel: string;
  OriginalCancelButtonEnabled: Boolean;
  OriginalCancelButtonModalResult: Integer;
  ctrl: TWinControl;
begin
  RemoveSettingsChecked := True;
  RemoveLogsChecked := True;

  if UninstallSilent then
    exit;

  ctrl := UninstallProgressForm.CancelButton;
  UninstallButton := TNewButton.Create(UninstallProgressForm);
  UninstallButton.Parent := UninstallProgressForm;
  UninstallButton.Left := ctrl.Left - ctrl.Width - ScaleX(10);
  UninstallButton.Top := ctrl.Top;
  UninstallButton.Width := ctrl.Width;
  UninstallButton.Height := ctrl.Height;
  UninstallButton.TabOrder := ctrl.TabOrder;
  UninstallButton.Caption := ExpandConstant('{cm:UninstallButtonCaption}');
  UninstallButton.ModalResult := mrOK;
  UninstallProgressForm.CancelButton.TabOrder := UninstallButton.TabOrder + 1;

  UninstallPage := TNewNotebookPage.Create(UninstallProgressForm);
  UninstallPage.Notebook := UninstallProgressForm.InnerNotebook;
  UninstallPage.Parent := UninstallProgressForm.InnerNotebook;
  UninstallPage.Align := alClient;
  UninstallProgressForm.InnerNotebook.ActivePage := UninstallPage;

  ctrl := UninstallProgressForm.StatusLabel;

  RemoveSettingsCheckBox := TNewCheckBox.Create(UninstallProgressForm);
  RemoveSettingsCheckBox.Parent := UninstallPage;
  RemoveSettingsCheckBox.Top := ctrl.Top;
  RemoveSettingsCheckBox.Left := ctrl.Left;
  RemoveSettingsCheckBox.Width := ctrl.Width;
  RemoveSettingsCheckBox.Height := ScaleY(23);
  RemoveSettingsCheckBox.Caption := ExpandConstant('{cm:UninstallRemoveSettings}');
  RemoveSettingsCheckBox.Checked := True;
  RemoveSettingsCheckBox.TabStop := False;

  RemoveLogsCheckBox := TNewCheckBox.Create(UninstallProgressForm);
  RemoveLogsCheckBox.Parent := UninstallPage;
  RemoveLogsCheckBox.Top := RemoveSettingsCheckBox.Top + RemoveSettingsCheckBox.Height + ScaleY(14);
  RemoveLogsCheckBox.Left := ctrl.Left;
  RemoveLogsCheckBox.Width := ctrl.Width;
  RemoveLogsCheckBox.Height := ScaleY(23);
  RemoveLogsCheckBox.Caption := ExpandConstant('{cm:UninstallRemoveLogs}');
  RemoveLogsCheckBox.Checked := True;
  RemoveLogsCheckBox.TabStop := False;

  OriginalPageNameLabel := UninstallProgressForm.PageNameLabel.Caption;
  OriginalPageDescriptionLabel := UninstallProgressForm.PageDescriptionLabel.Caption;
  OriginalCancelButtonEnabled := UninstallProgressForm.CancelButton.Enabled;
  OriginalCancelButtonModalResult := UninstallProgressForm.CancelButton.ModalResult;

  UninstallProgressForm.PageNameLabel.Caption := ExpandConstant('{cm:UninstallDataPageName}');
  UninstallProgressForm.PageDescriptionLabel.Caption := ExpandConstant('{cm:UninstallDataPageDescription}');
  UninstallProgressForm.CancelButton.Enabled := True;
  UninstallProgressForm.CancelButton.ModalResult := mrCancel;
  UninstallProgressForm.ActiveControl := UninstallButton;

  if UninstallProgressForm.ShowModal = mrCancel then
    Abort;

  RemoveSettingsChecked := RemoveSettingsCheckBox.Checked;
  RemoveLogsChecked := RemoveLogsCheckBox.Checked;

  UninstallButton.Visible := False;
  UninstallProgressForm.PageNameLabel.Caption := OriginalPageNameLabel;
  UninstallProgressForm.PageDescriptionLabel.Caption := OriginalPageDescriptionLabel;
  UninstallProgressForm.CancelButton.Enabled := OriginalCancelButtonEnabled;
  UninstallProgressForm.CancelButton.ModalResult := OriginalCancelButtonModalResult;
  UninstallProgressForm.InnerNotebook.ActivePage := UninstallProgressForm.InstallingPage;
end;



function SwitchHasValue(Name: string; Value: string): Boolean;
begin
  Result := CompareText(ExpandConstant('{param:' + Name + '}'), Value) = 0;
end;

function IsUpdating(): Boolean;
begin
  Result := SwitchHasValue('update', 'true') and WizardSilent();
end;

function GetInstallDir(Param: string): string;
begin
  if IsUpdating() then
    Result := ExpandConstant('{app}\install')
  else
    Result := ExpandConstant('{app}');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and IsUpdating() then
    SaveStringToFile(ExpandConstant('{app}\updates\versions.txt'), '{#Version}' + #13#10, True);
end;

procedure DisableAppDirInheritance();
var
  ResultCode: Integer;
  Permissions: string;
begin
  Permissions := '/grant:r "*S-1-5-18:(OI)(CI)F" /grant:r "*S-1-5-32-544:(OI)(CI)F" /grant:r "*S-1-5-11:(OI)(CI)RX" /grant:r "*S-1-5-32-545:(OI)(CI)RX"';
  Permissions := Permissions + Format(' /grant:r "*S-1-3-0:(OI)(CI)F" /grant:r "%s:(OI)(CI)F"', [GetUserNameString()]);
  Exec(ExpandConstant('{sys}\icacls.exe'), ExpandConstant('"{app}" /inheritancelevel:r ') + Permissions, '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

procedure Explode(var Dest: TArrayOfString; Text: String; Separator: String);
var
  i, p: Integer;
begin
  i := 0;
  repeat
    SetArrayLength(Dest, i + 1);
    p := Pos(Separator, Text);
    if p > 0 then begin
      Dest[i] := Copy(Text, 1, p - 1);
      Text := Copy(Text, p + Length(Separator), Length(Text));
      i := i + 1;
    end else begin
      Dest[i] := Text;
      Text := '';
    end;
  until Length(Text) = 0;
end;

function NeedsAddToPath(InstallDir: string): Boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', OrigPath) then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + InstallDir + ';', ';' + OrigPath + ';') = 0;
end;

function AddToPath(InstallDir: string): string;
var
  OrigPath: string;
begin
  RegQueryStringValue(HKCU, 'Environment', 'Path', OrigPath);
  if (Length(OrigPath) > 0) and (OrigPath[Length(OrigPath)] = ';') then
    Result := OrigPath + InstallDir
  else
    Result := OrigPath + ';' + InstallDir;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Path: string;
  InstallDir: string;
  Parts: TArrayOfString;
  NewPath: string;
  i: Integer;
begin
  if CurUninstallStep <> usUninstall then
    exit;
  DelTree(ExpandConstant('{app}\updates'), True, True, True);
  DelTree(ExpandConstant('{app}\install'), True, True, True);
  DelTree(ExpandConstant('{app}\old'), True, True, True);
  if RemoveSettingsChecked then
    DelTree(ExpandConstant('{userappdata}\StealCode'), True, True, True);
  if RemoveLogsChecked then
    DelTree(ExpandConstant('{localappdata}\StealCode'), True, True, True);
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Path) then
    exit;
  InstallDir := ExpandConstant('{app}');
  NewPath := '';
  Explode(Parts, Path, ';');
  for i := 0 to GetArrayLength(Parts) - 1 do begin
    if CompareText(Parts[i], InstallDir) <> 0 then begin
      NewPath := NewPath + Parts[i];
      if i < GetArrayLength(Parts) - 1 then
        NewPath := NewPath + ';';
    end;
  end;
  RegWriteExpandStringValue(HKCU, 'Environment', 'Path', NewPath);
end;
