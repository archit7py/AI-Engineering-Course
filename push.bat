@echo off
echo.
echo ===============================
echo      AI ENGINEERING PUSH
echo ===============================
echo.

git add .

set /p msg=Enter Commit Message: 

git commit -m "%msg%"

git push origin main

echo.
echo ===============================
echo     PUSH COMPLETED
echo ===============================
pause