@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo   MCP-Fabric Server Setup
echo ========================================
echo.

:: ----------------------------------------
:: Step 0: Check for Git
:: ----------------------------------------
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not in PATH.
    echo Please install Git from https://git-scm.com/downloads
    echo.
    pause
    exit /b 1
)

:: ----------------------------------------
:: Step 1: Check for Python 3.12+
:: ----------------------------------------
echo Checking for Python 3.12 or higher...
set "PYTHON_CMD="

:: First try py launcher with 3.13 then 3.12
py -3.13 --version >nul 2>&1
if !errorlevel!==0 (
    set "PYTHON_CMD=py -3.13"
    goto :python_found
)

py -3.12 --version >nul 2>&1
if !errorlevel!==0 (
    set "PYTHON_CMD=py -3.12"
    goto :python_found
)

:: Try common installation paths
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto :python_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :python_found
)
if exist "%PROGRAMFILES%\Python313\python.exe" (
    set "PYTHON_CMD=%PROGRAMFILES%\Python313\python.exe"
    goto :python_found
)
if exist "%PROGRAMFILES%\Python312\python.exe" (
    set "PYTHON_CMD=%PROGRAMFILES%\Python312\python.exe"
    goto :python_found
)
if exist "C:\Python313\python.exe" (
    set "PYTHON_CMD=C:\Python313\python.exe"
    goto :python_found
)
if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
    goto :python_found
)

:: Python 3.12+ not found - install automatically
echo Python 3.12+ not found. Installing Python 3.12 automatically...
echo.

:: Check if winget is available
winget --version >nul 2>&1
if %errorlevel%==0 (
    echo Installing Python 3.12 via winget...
    echo This may take a few minutes...
    echo.
    winget install Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent

    if errorlevel 1 (
        echo.
        echo WARNING: winget installation may have had issues. Trying alternative...
        goto :try_direct_download
    )

    echo.
    echo Python installed. Locating executable...
    timeout /t 2 /nobreak >nul

    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        goto :python_found
    )
    if exist "%PROGRAMFILES%\Python312\python.exe" (
        set "PYTHON_CMD=%PROGRAMFILES%\Python312\python.exe"
        goto :python_found
    )
    if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe" (
        set "PYTHON_CMD=%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
        goto :python_found
    )

    py -3.12 --version >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON_CMD=py -3.12"
        goto :python_found
    )

    echo.
    echo Python was installed but cannot be located.
    echo Please close this window, open a NEW terminal, and run setup again.
    echo.
    pause
    exit /b 1
)

:try_direct_download
echo winget not available. Downloading Python 3.12 installer...
echo.

set "installerUrl=https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe"
set "installerPath=%TEMP%\python-3.12.8-amd64.exe"

powershell -Command "Invoke-WebRequest -Uri '%installerUrl%' -OutFile '%installerPath%'" 2>nul

if not exist "%installerPath%" (
    echo.
    echo ERROR: Failed to download Python installer.
    echo Please install Python 3.12+ manually from:
    echo   https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Installing Python 3.12 (this may take a minute)...
start /wait "" "%installerPath%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0

del "%installerPath%" >nul 2>&1
timeout /t 3 /nobreak >nul

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :python_found
)
if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
    goto :python_found
)
if exist "%PROGRAMFILES%\Python312\python.exe" (
    set "PYTHON_CMD=%PROGRAMFILES%\Python312\python.exe"
    goto :python_found
)

py -3.12 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3.12"
    goto :python_found
)

echo.
echo Python was installed but cannot be located in this session.
echo Please close this window, open a NEW terminal, and run setup again.
echo.
pause
exit /b 1

:python_found
for /f "tokens=*" %%i in ('!PYTHON_CMD! --version') do echo Found: %%i
echo Using: !PYTHON_CMD!
echo.

:: ----------------------------------------
:: Step 2: Check for uv (recommended package manager)
:: ----------------------------------------
echo Checking for uv package manager...
set "UV_CMD="

uv --version >nul 2>&1
if !errorlevel!==0 (
    set "UV_CMD=uv"
    for /f "tokens=*" %%i in ('uv --version') do echo Found: %%i
    echo.
    goto :uv_done
)

:: uv not found - install it
echo uv not found. Installing uv (recommended by the project)...
echo.
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" 2>nul

:: Refresh PATH to find uv - check all possible install locations
set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv\bin;%USERPROFILE%\.cargo\bin;%CARGO_HOME%\bin;%PATH%"

:: Also check common Windows install locations directly
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV_CMD=%USERPROFILE%\.local\bin\uv.exe"
    echo uv installed successfully.
    echo.
    goto :uv_done
)
if exist "%LOCALAPPDATA%\uv\bin\uv.exe" (
    set "UV_CMD=%LOCALAPPDATA%\uv\bin\uv.exe"
    echo uv installed successfully.
    echo.
    goto :uv_done
)
if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    set "UV_CMD=%USERPROFILE%\.cargo\bin\uv.exe"
    echo uv installed successfully.
    echo.
    goto :uv_done
)

uv --version >nul 2>&1
if !errorlevel!==0 (
    set "UV_CMD=uv"
    echo uv installed successfully.
    echo.
    goto :uv_done
)

echo WARNING: Could not install uv. Will use pip as fallback.
echo.

:uv_done

:: ----------------------------------------
:: Step 3: Select clone location
:: ----------------------------------------
set "defaultPath=%USERPROFILE%\repos"
echo Where would you like to clone the repository?
echo.
echo Opening folder browser dialog...
echo (Default if cancelled: %defaultPath%)
echo.

:: Use PowerShell to show a folder browser dialog
for /f "usebackq delims=" %%i in (`powershell -ExecutionPolicy Bypass -Command ^
    "Add-Type -AssemblyName System.Windows.Forms;" ^
    "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog;" ^
    "$dialog.Description = 'Select folder to clone MCP-Fabric repository';" ^
    "$dialog.RootFolder = 'MyComputer';" ^
    "$dialog.SelectedPath = '%defaultPath%';" ^
    "$dialog.ShowNewFolderButton = $true;" ^
    "if ($dialog.ShowDialog() -eq 'OK') { $dialog.SelectedPath } else { '' }"`) do set "clonePath=%%i"

if "!clonePath!"=="" (
    echo No folder selected, using default: %defaultPath%
    set "clonePath=%defaultPath%"
) else (
    echo Selected folder: !clonePath!
)

:: Create directory if it doesn't exist
if not exist "!clonePath!" (
    echo.
    echo Creating directory: !clonePath!
    mkdir "!clonePath!"
    if errorlevel 1 (
        echo ERROR: Failed to create directory!
        pause
        exit /b 1
    )
)

set "repoPath=!clonePath!\MCP-Fabric"

:: Check if repo already exists
if exist "!repoPath!" (
    echo.
    echo Directory already exists: !repoPath!
    set /p "overwrite=Do you want to remove it and clone fresh? (y/N): "
    if /i "!overwrite!"=="y" (
        rmdir /s /q "!repoPath!"
    ) else (
        echo Skipping clone, using existing directory.
        goto :skip_clone
    )
)

:: ----------------------------------------
:: Step 4: Clone the repository
:: ----------------------------------------
echo.
echo [Step 1/5] Cloning repository...
echo   URL: https://github.com/bibiibjorn/MCP-Fabric.git
echo   To:  !repoPath!
echo.

cd /d "!clonePath!"
if errorlevel 1 (
    echo ERROR: Cannot access directory: !clonePath!
    pause
    exit /b 1
)

git clone https://github.com/bibiibjorn/MCP-Fabric.git

if errorlevel 1 (
    echo.
    echo ERROR: Failed to clone repository!
    echo Please check that the URL is accessible and you have internet.
    echo.
    pause
    exit /b 1
)

:skip_clone

:: Verify the repo directory exists before entering it
if not exist "!repoPath!" (
    echo.
    echo ERROR: Repository directory not found at: !repoPath!
    echo The clone may have failed or created a different folder name.
    echo.
    pause
    exit /b 1
)

cd /d "!repoPath!"
if errorlevel 1 (
    echo ERROR: Cannot enter directory: !repoPath!
    pause
    exit /b 1
)

:: Verify pyproject.toml exists (needed for dependency install)
if not exist "!repoPath!\pyproject.toml" (
    echo.
    echo ERROR: pyproject.toml not found in !repoPath!
    echo The repository may be incomplete or corrupted.
    echo.
    pause
    exit /b 1
)

echo.
echo Repository ready at: !repoPath!

:: ----------------------------------------
:: Step 5: Set up virtual environment and install dependencies
:: ----------------------------------------
set "venvDir="

if not defined UV_CMD goto :pip_install

:: Try uv first
echo.
echo [Step 2/5] Setting up with uv (virtual environment + dependencies)...
echo This may take a few minutes on first run...
echo.

!UV_CMD! sync

if errorlevel 1 (
    echo.
    echo WARNING: uv sync failed. Falling back to pip...
    echo.
    goto :pip_install
)

:: uv creates .venv by default
set "venvDir=.venv"
echo.
echo uv setup completed successfully.
goto :deps_done

:pip_install
:: Fallback: use pip with venv
echo.
echo [Step 2/5] Creating virtual environment with pip...
!PYTHON_CMD! -m venv venv

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create virtual environment!
    echo.
    pause
    exit /b 1
)

if not exist "!repoPath!\venv\Scripts\python.exe" (
    echo.
    echo ERROR: Virtual environment was not created properly!
    echo The venv folder or python.exe is missing.
    echo.
    pause
    exit /b 1
)

set "venvDir=venv"
echo Virtual environment created successfully.

:: Activate and install
echo.
echo [Step 3/5] Activating virtual environment...
call "!repoPath!\venv\Scripts\activate.bat"

echo.
echo [Step 4/5] Installing dependencies from pyproject.toml...
echo This may take a few minutes...
echo.

pip install -e "!repoPath!"

if errorlevel 1 (
    echo.
    echo WARNING: Some dependencies may have failed to install.
    echo You may need to resolve these manually after setup.
    echo.
    echo Press any key to continue anyway...
    pause >nul
)

:deps_done

:: Determine the correct venv python path
if "!venvDir!"==".venv" (
    set "venvPython=!repoPath!\.venv\Scripts\python.exe"
) else (
    set "venvPython=!repoPath!\venv\Scripts\python.exe"
)

:: Verify python exists in venv
if not exist "!venvPython!" (
    echo.
    echo ERROR: Python executable not found at: !venvPython!
    echo Virtual environment may not have been created properly.
    echo.
    pause
    exit /b 1
)

echo.
echo Virtual environment Python: !venvPython!

:: ----------------------------------------
:: Step 6: Configure Claude Desktop
:: ----------------------------------------
echo.
echo ========================================
echo   Claude Desktop Configuration
echo ========================================
echo.

set "configPath=%APPDATA%\Claude\claude_desktop_config.json"
echo Detected config path: !configPath!

:: Check if config directory exists
for %%F in ("!configPath!") do set "configDir=%%~dpF"
if not exist "!configDir!" (
    echo Creating config directory: !configDir!
    mkdir "!configDir!"
)

echo.
echo [Step 5/5] Updating Claude Desktop config...

:: Use PowerShell to handle JSON manipulation properly
powershell -ExecutionPolicy Bypass -Command ^
    "$configPath = '!configPath!'; $repoPath = '!repoPath!';" ^
    "$venvPython = '!venvPython!';" ^
    "$serverName = 'MCP-Fabric';" ^
    "$scriptPath = Join-Path $repoPath 'fabric_mcp.py';" ^
    "$mcpServer = @{ 'command' = $venvPython; 'args' = @($scriptPath) };" ^
    "if (Test-Path $configPath) { try { $config = Get-Content $configPath -Raw -Encoding UTF8 | ConvertFrom-Json; Write-Host 'Found existing config file' -ForegroundColor Green } catch { Write-Host 'Config file exists but is invalid, creating new one' -ForegroundColor Yellow; $config = [PSCustomObject]@{} } } else { Write-Host 'Creating new config file' -ForegroundColor Yellow; $config = [PSCustomObject]@{} };" ^
    "if (-not $config.PSObject.Properties['mcpServers']) { $config | Add-Member -NotePropertyName 'mcpServers' -NotePropertyValue ([PSCustomObject]@{}) };" ^
    "if ($config.mcpServers.PSObject.Properties[$serverName]) { $config.mcpServers.$serverName = $mcpServer; Write-Host ('Updated existing ' + $serverName + ' configuration') -ForegroundColor Green } else { $config.mcpServers | Add-Member -NotePropertyName $serverName -NotePropertyValue $mcpServer; Write-Host ('Added ' + $serverName + ' configuration') -ForegroundColor Green };" ^
    "$json = $config | ConvertTo-Json -Depth 10; [System.IO.File]::WriteAllText($configPath, $json, [System.Text.UTF8Encoding]::new($false));" ^
    "Write-Host ''; Write-Host 'Config saved to:' $configPath -ForegroundColor Cyan;" ^
    "Write-Host ''; Write-Host 'MCP Server configured as:' $serverName -ForegroundColor Cyan;" ^
    "Write-Host '  Python: ' $venvPython; Write-Host '  Script: ' $scriptPath;"

if errorlevel 1 (
    echo.
    echo Warning: Failed to update Claude Desktop config automatically.
    echo You may need to add the following entry to mcpServers in !configPath!:
    echo.
    echo     "MCP-Fabric": {
    echo       "command": "!venvPython!",
    echo       "args": ["!repoPath!\fabric_mcp.py"]
    echo     }
    echo.
)

:: ----------------------------------------
:: Success message
:: ----------------------------------------
echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Repository cloned to: !repoPath!
echo Virtual environment:  !repoPath!\!venvDir!
echo Claude config:        !configPath!
echo MCP Server added as:  MCP-Fabric
echo.
echo NOTE: Your existing MCP servers are preserved.
echo.
echo IMPORTANT: Restart Claude Desktop for changes to take effect!
echo.
echo To start working manually:
echo   1. cd "!repoPath!"
if "!venvDir!"==".venv" (
    echo   2. .venv\Scripts\activate.bat
) else (
    echo   2. venv\Scripts\activate.bat
)
echo   3. python fabric_mcp.py
echo.

:: Keep the window open
pause
