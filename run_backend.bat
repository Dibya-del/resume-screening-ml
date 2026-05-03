@echo off
cd /d C:\ML_PROJECTS\resume_screening_ml\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
