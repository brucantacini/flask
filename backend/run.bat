@echo off
echo Ativando ambiente virtual...
call venv\Scripts\activate

echo Iniciando servidor Flask...
python server.py

pause