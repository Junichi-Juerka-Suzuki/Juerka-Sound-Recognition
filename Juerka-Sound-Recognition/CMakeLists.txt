# CMakeList.txt : Juerka-Sound-Recognition の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。
#

# ソースをこのプロジェクトの実行可能ファイルに追加します。
add_executable (Juerka-Sound-Recognition "Juerka-Sound-Recognition.cpp" "Juerka-Sound-Recognition.h")

target_link_libraries(Juerka-Sound-Recognition Juerka-core)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Juerka-Sound-Recognition PROPERTY CXX_STANDARD 20)
endif()

# TODO: テストを追加し、必要な場合は、ターゲットをインストールします。
