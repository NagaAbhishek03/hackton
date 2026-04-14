@echo off
echo Running Git commands to upload your project to GitHub...

git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/NagaAbhishek03/hackton.git
git push -u origin main

echo.
echo Process complete! Please check if there are any errors above.
pause
