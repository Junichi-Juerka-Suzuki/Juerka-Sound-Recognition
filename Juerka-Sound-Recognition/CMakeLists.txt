# CMakeList.txt : Juerka-Sound-Recognition の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。
#

# ソースをこのプロジェクトの実行可能ファイルに追加します。
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

add_executable(Juerka-Sound-Recognition-Voice-Recognition-Test
	"test/voice_recognition_test.cpp" "CurrentGenerator/Fft.cpp"
	"CurrentGenerator/Fft.h" "CurrentGenerator/SoundDataManager.h"
	"CurrentGenerator/Common.h" "CurrentGenerator/SoundDataManager.cpp"
	"CurrentGenerator/SoundCurrentGenerator.h")

target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test PUBLIC "../Juerka/CommonNet")
target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test PUBLIC "../Juerka/Utility")
target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test PUBLIC "CurrentGenerator")

set_property(TARGET Juerka-Sound-Recognition-Voice-Recognition-Test PROPERTY CXX_STANDARD 20)

target_link_libraries(Juerka-Sound-Recognition-Voice-Recognition-Test PUBLIC Juerka-core MKL::MKL)
