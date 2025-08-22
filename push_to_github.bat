@echo off
REM Script to push changes to GitHub repository

echo Preparing to push AEGIS changes to GitHub...

REM Add all changes
git add .

REM Commit changes with a descriptive message
git commit -m "feat: Complete AEGIS implementation with Gradio web interface

- Implemented complete AEGIS system with 9 AI alignment risk categories
- Added 45+ attack vectors based on peer-reviewed research
- Created comprehensive evaluation framework
- Developed Gradio web interface for easy access
- Added user guide and documentation
- Fixed all critical issues identified in review
- Implemented backward compatibility API
- Added quick start examples and test suite"

REM Push to GitHub
git push origin main

echo Changes pushed to GitHub successfully!
pause