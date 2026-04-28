#if WINDOWS

using System.Runtime.InteropServices;


public static class ConsoleHelper
{
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint GetConsoleProcessList(uint[] processList, uint processCount);

    public static bool IsLaunchedFromExplorer()
    {
        // This logic is primarily for Windows
        if(!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return false;

        uint[] processList = new uint[1];
        uint count = GetConsoleProcessList(processList, 1);

        // If count is 1, it means this process is the only one attached to the console
        // (i.e., the console was created specifically for this process)
        return count <= 1;
    }
}
#endif