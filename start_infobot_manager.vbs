Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
venvPython = fso.BuildPath(projectDir, ".venv\Scripts\pythonw.exe")
launcher = fso.BuildPath(projectDir, "launcher\infobot_manager.py")

If fso.FileExists(venvPython) Then
    cmd = """" & venvPython & """ """ & launcher & """"
Else
    cmd = "pythonw """ & launcher & """"
End If

shell.CurrentDirectory = projectDir
shell.Run cmd, 0, False

