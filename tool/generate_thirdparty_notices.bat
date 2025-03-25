@echo off

rem #####AIT Dockerイメージ作成#####

set DOCKER_IMAGE_NAME=ait_license_base:latest

cd /d %~dp0

rem 既存削除
echo start docker crean up...
docker rmi %DOCKER_IMAGE_NAME%
docker system prune -f

rem ビルド
echo start docker build...
docker build -f ..\deploy\container\dockerfile -t %DOCKER_IMAGE_NAME% ..\deploy\container

rem #####AITライセンス情報出力#####

set LICENSE_DOCKER_IMAGE_NAME=ait_license_thirdparty_notices

cd /d %~dp0

rem 既存削除
echo start docker crean up...
docker rmi %LICENSE_DOCKER_IMAGE_NAME%
docker system prune -f

rem ビルド
echo start docker build...
docker build -t %LICENSE_DOCKER_IMAGE_NAME% -f ..\deploy\container\dockerfile_license ..\deploy\container

rem 実行
echo run docker...
docker run %LICENSE_DOCKER_IMAGE_NAME%:latest > ../ThirdPartyNotices.txt

pause