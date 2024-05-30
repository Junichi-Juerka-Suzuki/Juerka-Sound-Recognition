# CMakeList.txt : Juerka-Sound-Recognition の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。
#

# ソースをこのプロジェクトの実行可能ファイルに追加します。
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

add_executable(Juerka-Sound-Recognition-Voice-Recognition-Test-1
	"test/voice_recognition_test_1.cpp" "CurrentGenerator/Fft.cpp"
	"CurrentGenerator/Fft.h" "CurrentGenerator/SoundDataManager.h"
	"CurrentGenerator/Common.h" "CurrentGenerator/SoundDataManager.cpp"
	"CurrentGenerator/SoundCurrentGenerator.h")

target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test-1 PUBLIC "../Juerka/CommonNet")
target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test-1 PUBLIC "../Juerka/Utility")
target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test-1 PUBLIC "CurrentGenerator")

set_property(TARGET Juerka-Sound-Recognition-Voice-Recognition-Test-1 PROPERTY CXX_STANDARD 20)

target_link_libraries(Juerka-Sound-Recognition-Voice-Recognition-Test-1 PUBLIC Juerka-core MKL::MKL)

add_executable(Juerka-Sound-Recognition-Voice-Recognition-Test-2
	"test/voice_recognition_test_2.cpp" "CurrentGenerator/Fft.cpp"
	"CurrentGenerator/Fft.h" "CurrentGenerator/SoundDataManager.h"
	"CurrentGenerator/Common.h" "CurrentGenerator/SoundDataManager.cpp"
	"CurrentGenerator/SoundCurrentGenerator.h")

target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test-2 PUBLIC "../Juerka/CommonNet")
target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test-2 PUBLIC "../Juerka/Utility")
target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test-2 PUBLIC "CurrentGenerator")

set_property(TARGET Juerka-Sound-Recognition-Voice-Recognition-Test-2 PROPERTY CXX_STANDARD 20)

target_link_libraries(Juerka-Sound-Recognition-Voice-Recognition-Test-2 PUBLIC Juerka-core MKL::MKL)

add_executable(Juerka-Sound-Recognition-Voice-Recognition-Test-3
	"test/voice_recognition_test_3.cpp" "CurrentGenerator/Fft.cpp"
	"CurrentGenerator/Fft.h" "CurrentGenerator/SoundDataManager.h"
	"CurrentGenerator/Common.h" "CurrentGenerator/SoundDataManager.cpp"
	"CurrentGenerator/SoundCurrentGenerator.h")

target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test-3 PUBLIC "../Juerka/CommonNet")
target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test-3 PUBLIC "../Juerka/Utility")
target_include_directories(Juerka-Sound-Recognition-Voice-Recognition-Test-3 PUBLIC "CurrentGenerator")

set_property(TARGET Juerka-Sound-Recognition-Voice-Recognition-Test-3 PROPERTY CXX_STANDARD 20)

target_link_libraries(Juerka-Sound-Recognition-Voice-Recognition-Test-3 PUBLIC Juerka-core MKL::MKL)

