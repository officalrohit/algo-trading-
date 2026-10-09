Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
strDir = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = strDir

' Launch Streamlit in completely hidden background mode (0 = hide window)
WshShell.Run "cmd /c python -m streamlit run app.py --server.headless true --server.port 8501", 0, False

' Wait 2 seconds for server initialization and open browser
WScript.Sleep 2500
WshShell.Run "http://localhost:8501"
