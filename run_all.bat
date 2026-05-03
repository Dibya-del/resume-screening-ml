@echo off
start "Resume Screening Backend" cmd /k "cd /d C:\ML_PROJECTS\resume_screening_ml\backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000"
start "Resume Screening Frontend" cmd /k "cd /d C:\ML_PROJECTS\resume_screening_ml\frontend && npm run dev"
