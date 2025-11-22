Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
venvPython = fso.BuildPath(projectDir, ".venv\Scripts\pythonw.exe")
script = fso.BuildPath(projectDir, "scripts\git_commit_gui.py")

If fso.FileExists(venvPython) Then
    cmd = """" & venvPython & """ """ & script & """"
Else
    cmd = "pythonw """ & script & """"
End If

shell.CurrentDirectory = projectDir
shell.Run cmd, 0, False

