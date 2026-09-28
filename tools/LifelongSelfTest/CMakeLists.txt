# HERoEHS lifelong 코어 자체 시험 — rtabmap 트리 빌드(tools/CMakeLists.txt)에 끼우지 않은 독립 프로젝트라
# 이미지 굽기에 영향이 없다. 코어가 깔린 환경(이미지 안)에서:
#   cmake -S <이 디렉토리> -B /tmp/lst && cmake --build /tmp/lst -j4 && /tmp/lst/rtabmap-lifelong_selftest /tmp/lst_run
# 설치 안 된 빌드 트리로 돌릴 땐 -DRTABMap_DIR=<코어 빌드 디렉토리>.
cmake_minimum_required(VERSION 3.14)
project(lifelong_selftest C CXX)   # C 필수: PCL→VTK 가 MPI::MPI_C 를 찾는다
set(CMAKE_CXX_STANDARD 17)
find_package(RTABMap REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(SQLite3 REQUIRED)
add_executable(lifelong_selftest main.cpp)
target_link_libraries(lifelong_selftest rtabmap::core rtabmap::utilite ${OpenCV_LIBS} SQLite::SQLite3)
set_target_properties(lifelong_selftest PROPERTIES OUTPUT_NAME rtabmap-lifelong_selftest)
