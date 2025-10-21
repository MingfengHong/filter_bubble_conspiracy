@echo off
echo ============================================================
echo 信息茧房与确认偏误共谋模型 - Solara 可视化服务器
echo ============================================================
echo.
echo 正在启动服务器...
echo 服务器启动后将在浏览器中打开
echo 或手动访问: http://localhost:8765
echo.
echo 按 Ctrl+C 停止服务器
echo ============================================================
echo.

C:\ProgramData\anaconda3\envs\mesa_conspiracy\python.exe -m solara run app.py --host localhost --port 8765

pause


