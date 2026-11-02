@echo off
setlocal
set "VAS_VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if not exist "%VAS_VSDEVCMD%" (
  echo Visual Studio 2022 Build Tools not found.
  exit /b 1
)
call "%VAS_VSDEVCMD%" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%
%*
