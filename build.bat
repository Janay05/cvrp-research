call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
mkdir build2
cl.exe /O2 /EHsc /std:c++17 /Fe:build2\cvrp_parallel.exe src\*.cpp
