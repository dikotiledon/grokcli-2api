@echo off
REM Go main process. Prefer WSL/Docker for full sidecar stack.
REM Local Windows path: build (if needed) -> migrate up -> start binary.
cd /d %~dp0

REM Load .env into process environment (KEY=VALUE lines only).
if exist .env (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" if not "%%B"=="" set "%%A=%%B"
  )
)

if not exist bin\grok2api.exe (
  echo Building bin\grok2api.exe ...
  go build -o bin\grok2api.exe .\cmd\grok2api
  if errorlevel 1 exit /b 1
)

REM Auto-migrate default ON (set GROK2API_AUTO_MIGRATE=0 to skip)
if /I "%GROK2API_AUTO_MIGRATE%"=="0" goto :skip_migrate
if /I "%GROK2API_AUTO_MIGRATE%"=="false" goto :skip_migrate
if /I "%GROK2API_AUTO_MIGRATE%"=="no" goto :skip_migrate
if /I "%GROK2API_AUTO_MIGRATE%"=="off" goto :skip_migrate

if not exist bin\grok2api-migrate.exe (
  echo Building bin\grok2api-migrate.exe ...
  go build -o bin\grok2api-migrate.exe .\cmd\grok2api-migrate
  if errorlevel 1 exit /b 1
)
if "%GROK2API_MIGRATIONS_DIR%"=="" set "GROK2API_MIGRATIONS_DIR=migrations"
if not exist "%GROK2API_MIGRATIONS_DIR%" (
  echo ERROR: migrations directory missing: %GROK2API_MIGRATIONS_DIR%
  exit /b 1
)
echo Applying migrations (dir=%GROK2API_MIGRATIONS_DIR%) ...
bin\grok2api-migrate.exe -dir "%GROK2API_MIGRATIONS_DIR%" up
if errorlevel 1 exit /b 1
bin\grok2api-migrate.exe -dir "%GROK2API_MIGRATIONS_DIR%" verify
if errorlevel 1 exit /b 1
goto :start_app

:skip_migrate
echo GROK2API_AUTO_MIGRATE=%GROK2API_AUTO_MIGRATE%; skip schema migrate

:start_app
set GROK2API_RUNTIME=go
set GROK2API_GO_PUBLIC_READ=1
set GROK2API_GO_CHAT=1
set GROK2API_GO_MESSAGES=1
set GROK2API_GO_RESPONSES=1
set GROK2API_GO_ADMIN_READ=1
set GROK2API_GO_ADMIN_WRITE=1
set GROK2API_GO_MAINTAINER=1
set GROK2API_GO_WRITES=1
bin\grok2api.exe
