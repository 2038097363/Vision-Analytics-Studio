param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string]$ProcessName = "vision-analytics-studio"
)

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class NativeFileDialog {
    private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumWindowsProc proc, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetClassName(IntPtr hWnd, StringBuilder className, int maxCount);

    [DllImport("user32.dll")]
    private static extern IntPtr GetDlgItem(IntPtr hDlg, int id);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr SendMessage(IntPtr hWnd, uint message, IntPtr wParam, string lParam);

    [DllImport("user32.dll")]
    private static extern IntPtr SendMessage(IntPtr hWnd, uint message, IntPtr wParam, IntPtr lParam);

    public static IntPtr FindForProcess(uint targetProcessId) {
        IntPtr result = IntPtr.Zero;
        EnumWindows((window, _) => {
            uint processId;
            GetWindowThreadProcessId(window, out processId);
            if (processId != targetProcessId) return true;
            var className = new StringBuilder(64);
            GetClassName(window, className, className.Capacity);
            if (className.ToString() == "#32770") {
                result = window;
                return false;
            }
            return true;
        }, IntPtr.Zero);
        return result;
    }

    public static bool SelectFile(IntPtr dialog, string path) {
        IntPtr fileName = GetDlgItem(dialog, 1148);
        IntPtr openButton = GetDlgItem(dialog, 1);
        if (fileName == IntPtr.Zero || openButton == IntPtr.Zero) return false;
        SendMessage(fileName, 0x000C, IntPtr.Zero, path);
        SendMessage(openButton, 0x00F5, IntPtr.Zero, IntPtr.Zero);
        return true;
    }
}
"@

$process = Get-Process -Name $ProcessName -ErrorAction Stop | Select-Object -First 1
$dialog = [IntPtr]::Zero
for ($attempt = 0; $attempt -lt 30 -and $dialog -eq [IntPtr]::Zero; $attempt++) {
    $dialog = [NativeFileDialog]::FindForProcess([uint32]$process.Id)
    if ($dialog -eq [IntPtr]::Zero) {
        Start-Sleep -Milliseconds 100
    }
}

if ($dialog -eq [IntPtr]::Zero) {
    throw "The application file dialog did not appear."
}

$absolutePath = [System.IO.Path]::GetFullPath($Path)
if (-not [NativeFileDialog]::SelectFile($dialog, $absolutePath)) {
    throw "The file dialog controls were not available."
}
