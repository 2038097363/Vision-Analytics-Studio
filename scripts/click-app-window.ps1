param(
    [int]$X,
    [int]$Y,
    [string]$ProcessName = "vision-analytics-studio"
)

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class AppWindowInput {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int x, int y);

    [DllImport("user32.dll")]
    public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extraInfo);

    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();
}
"@

[AppWindowInput]::SetProcessDPIAware() | Out-Null

$process = Get-Process -Name $ProcessName -ErrorAction Stop |
    Where-Object { $_.MainWindowHandle -ne 0 } |
    Select-Object -First 1

if (-not $process) {
    throw "No visible window was found for process '$ProcessName'."
}

$rect = New-Object AppWindowInput+RECT
if (-not [AppWindowInput]::GetWindowRect($process.MainWindowHandle, [ref]$rect)) {
    throw "Could not read the application window bounds."
}

[AppWindowInput]::SetForegroundWindow($process.MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 250
[AppWindowInput]::SetCursorPos($rect.Left + $X, $rect.Top + $Y) | Out-Null
[AppWindowInput]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
[AppWindowInput]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
