Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
venvPython = fso.BuildPath(projectDir, ".venv\Scripts\pythonw.exe")
launcher = fso.BuildPath(projectDir, "launcher\infobot_manager.py")

' Функция проверки наличия команды
Function CommandExists(cmd)
    On Error Resume Next
    Dim exitCode
    If InStr(cmd, " ") > 0 Then
        ' Команда с параметрами - используем cmd /c
        exitCode = shell.Run("cmd /c " & cmd & " --version >nul 2>&1", 0, True)
    Else
        ' Простая команда
        exitCode = shell.Run("cmd /c """ & cmd & """ --version >nul 2>&1", 0, True)
    End If
    CommandExists = (exitCode = 0)
    On Error GoTo 0
End Function

' Функция проверки версии Python (должна быть >= 3.13)
Function CheckPythonVersion()
    On Error Resume Next
    Dim versionOutput, versionParts, major, minor
    Dim pythonCmd
    
    ' Пробуем разные команды для получения версии
    If CommandExists("python") Then
        pythonCmd = "python"
    ElseIf CommandExists("py") Then
        pythonCmd = "py -3"
    ElseIf CommandExists("python3") Then
        pythonCmd = "python3"
    Else
        CheckPythonVersion = False
        Exit Function
    End If
    
    ' Получаем версию Python
    Dim wshExec
    Set wshExec = shell.Exec("cmd /c " & pythonCmd & " --version")
    versionOutput = wshExec.StdOut.ReadAll
    Set wshExec = Nothing
    
    ' Парсим версию (формат: Python 3.13.0)
    If InStr(versionOutput, "Python") > 0 Then
        versionParts = Split(versionOutput, " ")
        If UBound(versionParts) >= 1 Then
            Dim versionStr
            versionStr = versionParts(1)
            versionParts = Split(versionStr, ".")
            If UBound(versionParts) >= 1 Then
                major = CInt(versionParts(0))
                minor = CInt(versionParts(1))
                ' Проверяем версию >= 3.13
                If major > 3 Or (major = 3 And minor >= 13) Then
                    CheckPythonVersion = True
                    Exit Function
                End If
            End If
        End If
    End If
    
    CheckPythonVersion = False
    On Error GoTo 0
End Function

' Проверка Python
pythonFound = False
If CommandExists("python") Or CommandExists("py") Or CommandExists("python3") Then
    If CheckPythonVersion() Then
        pythonFound = True
    End If
End If

If Not pythonFound Then
    ' Попытка установки через winget
    wingetFound = CommandExists("winget")
    If wingetFound Then
        ' Тихая установка Python 3.13 через winget
        shell.Run "winget install --id Python.Python.3.13 --silent --accept-package-agreements --accept-source-agreements", 0, True
        ' Проверка после установки
        WScript.Sleep 3000
        If CheckPythonVersion() Then
            pythonFound = True
        End If
    End If
    
    ' Если Python всё ещё не найден или версия < 3.13 - открываем страницу скачивания
    If Not pythonFound Then
        shell.Run "https://www.python.org/downloads/windows/", 1, False
        WScript.Quit 1
    End If
End If

' Функция проверки установки Git через стандартные пути
Function GitInstalled()
    On Error Resume Next
    Dim gitPaths
    gitPaths = Array("C:\Program Files\Git\cmd\git.exe", "C:\Program Files (x86)\Git\cmd\git.exe", "C:\Program Files\Git\bin\git.exe")
    Dim i
    For i = 0 To UBound(gitPaths)
        If fso.FileExists(gitPaths(i)) Then
            GitInstalled = True
            Exit Function
        End If
    Next
    ' Проверяем через команду git
    If CommandExists("git") Then
        GitInstalled = True
        Exit Function
    End If
    GitInstalled = False
    On Error GoTo 0
End Function

' Функция получения пути к Git для добавления в PATH
Function GetGitPath()
    On Error Resume Next
    Dim gitPaths
    gitPaths = Array("C:\Program Files\Git\cmd", "C:\Program Files (x86)\Git\cmd", "C:\Program Files\Git\bin")
    Dim i
    For i = 0 To UBound(gitPaths)
        If fso.FolderExists(gitPaths(i)) Then
            GetGitPath = gitPaths(i)
            Exit Function
        End If
    Next
    GetGitPath = ""
    On Error GoTo 0
End Function

' Проверка Git (только если Python установлен)
' Сначала проверяем, установлен ли Git
Dim gitCheckResult
gitCheckResult = GitInstalled()
If Not gitCheckResult Then
    ' Git не найден - пытаемся установить
    wingetFound = CommandExists("winget")
    If wingetFound Then
        ' Выводим сообщение об установке Git (показываем на 3 секунды)
        shell.Run "cmd /c echo [INFO] Установка Git через winget... && timeout /t 3 >nul", 1, True
        ' Тихая установка Git с максимальной интеграцией
        ' Параметры установщика Git передаются через --override
        ' /VERYSILENT /NORESTART /NOCANCEL /SP- /SUPPRESSMSGBOXES
        ' /COMPONENTS=icons,ext\shellhere,assoc,assoc_sh /PATHOPTION=user /EDITOR=nano
        Dim gitParams
        gitParams = "/VERYSILENT /NORESTART /NOCANCEL /SP- /SUPPRESSMSGBOXES /COMPONENTS=icons,ext\shellhere,assoc,assoc_sh /PATHOPTION=user /EDITOR=nano"
        ' Запускаем установку Git (скрыто, но ждем завершения)
        shell.Run "winget install --id Git.Git --silent --accept-package-agreements --accept-source-agreements --override """ & gitParams & """", 0, True
        ' Ждем завершения установки (winget может установить Git в фоне)
        WScript.Sleep 12000
        ' Проверяем установку через стандартные пути
        If Not GitInstalled() Then
            ' Если Git все еще не найден, ждем еще немного
            WScript.Sleep 5000
            ' Повторная проверка
            If Not GitInstalled() Then
                ' Последняя попытка - проверяем через команду git
                WScript.Sleep 3000
            End If
        End If
    End If
End If

' Безопасная инициализация Git репозитория (если Git установлен)
If GitInstalled() Then
    ' Настраиваем Git пользователя (если не настроен) - ДО любых операций
    Dim userCheck
    userCheck = shell.Run("cmd /c cd /d """ & projectDir & """ && git config user.name >nul 2>&1", 0, True)
    If userCheck <> 0 Then
        shell.Run "cmd /c cd /d """ & projectDir & """ && git config user.name ""InfoBot User""", 0, True
    End If
    userCheck = shell.Run("cmd /c cd /d """ & projectDir & """ && git config user.email >nul 2>&1", 0, True)
    If userCheck <> 0 Then
        shell.Run "cmd /c cd /d """ & projectDir & """ && git config user.email ""infobot@local""", 0, True
    End If
    
    Dim gitDir
    gitDir = fso.BuildPath(projectDir, ".git")
    If Not fso.FolderExists(gitDir) Then
        ' Инициализируем репозиторий БЕЗ pull/fetch, чтобы не перезаписать существующие файлы
        shell.Run "cmd /c cd /d """ & projectDir & """ && git init", 0, True
        shell.Run "cmd /c cd /d """ & projectDir & """ && git branch -m main", 0, True
        ' Добавляем remote с HTTPS URL
        shell.Run "cmd /c cd /d """ & projectDir & """ && git remote add origin https://github.com/HellEvro/TPM_Public.git", 0, True
    Else
        ' Репозиторий уже существует - проверяем и исправляем remote URL
        Dim remoteCheck
        remoteCheck = shell.Run("cmd /c cd /d """ & projectDir & """ && git remote get-url origin >nul 2>&1", 0, True)
        If remoteCheck = 0 Then
            ' Remote существует - проверяем, используется ли SSH
            Dim remoteUrlCmd
            Set remoteUrlCmd = shell.Exec("cmd /c cd /d """ & projectDir & """ && git remote get-url origin")
            Do While remoteUrlCmd.Status = 0
                WScript.Sleep 10
            Loop
            Dim currentUrl
            currentUrl = ""
            Do While Not remoteUrlCmd.StdOut.AtEndOfStream
                currentUrl = currentUrl & remoteUrlCmd.StdOut.ReadLine
            Loop
            If InStr(currentUrl, "git@github.com") > 0 Then
                ' Используется SSH - меняем на HTTPS
                shell.Run "cmd /c cd /d """ & projectDir & """ && git remote set-url origin https://github.com/HellEvro/TPM_Public.git", 0, True
            End If
        Else
            ' Remote не существует - добавляем
            shell.Run "cmd /c cd /d """ & projectDir & """ && git remote add origin https://github.com/HellEvro/TPM_Public.git", 0, True
        End If
    End If
    
    ' Делаем первый коммит, если нет коммитов (независимо от того, новый репозиторий или существующий)
    Dim commitCheck
    commitCheck = shell.Run("cmd /c cd /d """ & projectDir & """ && git rev-list --count HEAD >nul 2>&1", 0, True)
    If commitCheck <> 0 Then
        ' Нет коммитов - делаем первый коммит
        ' Добавляем все файлы
        shell.Run "cmd /c cd /d """ & projectDir & """ && git add -A", 0, True
        ' Делаем коммит
        shell.Run "cmd /c cd /d """ & projectDir & """ && git commit -m ""Initial commit: InfoBot Public repository""", 0, True
    End If
End If

' Обновляем PATH для текущей сессии, если Git установлен
Dim gitPath
gitPath = GetGitPath()
If gitPath <> "" Then
    ' Добавляем путь к Git в PATH через переменные окружения
    Dim envPath
    envPath = shell.ExpandEnvironmentStrings("%PATH%")
    If InStr(envPath, gitPath) = 0 Then
        ' Обновляем PATH в реестре для текущего пользователя (для будущих сессий)
        Dim wshShell
        Set wshShell = CreateObject("WScript.Shell")
        On Error Resume Next
        Dim userPath
        userPath = wshShell.RegRead("HKCU\Environment\PATH")
        If Err.Number = 0 Then
            If InStr(userPath, gitPath) = 0 Then
                wshShell.RegWrite "HKCU\Environment\PATH", userPath & ";" & gitPath, "REG_EXPAND_SZ"
            End If
        End If
        On Error GoTo 0
        Set wshShell = Nothing
    End If
End If

' Запуск приложения
If fso.FileExists(venvPython) Then
    cmd = """" & venvPython & """ """ & launcher & """"
Else
    ' Определяем Python для запуска
    Dim pythonCmd
    pythonCmd = ""
    If CommandExists("python") Then
        ' Проверяем наличие pythonw
        If shell.Run("cmd /c pythonw --version >nul 2>&1", 0, True) = 0 Then
            pythonCmd = "pythonw"
        Else
            pythonCmd = "python"
        End If
    ElseIf CommandExists("py") Then
        pythonCmd = "py -3"
    ElseIf CommandExists("python3") Then
        pythonCmd = "python3"
    End If
    
    If pythonCmd <> "" Then
        cmd = pythonCmd & " """ & launcher & """"
    Else
        ' Fallback
        cmd = "python """ & launcher & """"
    End If
End If

' Запускаем Python с обновленным PATH
shell.CurrentDirectory = projectDir
If gitPath <> "" Then
    ' Создаем временный bat файл с обновленным PATH
    Dim batFile
    batFile = fso.BuildPath(projectDir, "start_launcher_temp.bat")
    Dim batContent
    batContent = "@echo off" & vbCrLf
    batContent = batContent & "set PATH=%PATH%;" & gitPath & vbCrLf
    batContent = batContent & "cd /d """ & projectDir & """" & vbCrLf
    batContent = batContent & cmd & vbCrLf
    batContent = batContent & "del ""%~f0""" & vbCrLf
    
    Dim batStream
    Set batStream = fso.CreateTextFile(batFile, True)
    batStream.Write batContent
    batStream.Close
    Set batStream = Nothing
    
    ' Запускаем bat файл
    shell.Run """" & batFile & """", 0, False
Else
    ' Запускаем напрямую, если Git не установлен
    shell.Run cmd, 0, False
End If
