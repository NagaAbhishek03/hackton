@echo off
echo Running Git commands to upload your newly added files to GitHub...

git add .
git commit -m "added model files"
git push

echo.
echo Process complete! Please check if there are any errors above.
pause
