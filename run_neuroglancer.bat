@echo off
set CONDAPATH=C://Users//3i//anaconda3
set ENVNAME=tissuecyte
call %CONDAPATH%//Scripts//activate.bat %ENVNAME%
python gui_ng.py
pause