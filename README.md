# Лабораторна №4 — CUDA-обробка відео

Програма читає MP4 через OpenCV, передає послідовні кадри в пам'ять NVIDIA GPU, обчислює їх абсолютну попарну різницю через `cv::cuda::absdiff` та записує MP4-результат через OpenCV. Нульовий кадр результату чорний, адже для нього немає попереднього кадру. CPU fallback навмисно відсутній.

## Вхідні дані

`data/input/16756162_1280_720_25fps.mp4` — наданий ролик 1280×720, 25 FPS.

## Вимоги

- NVIDIA GPU, драйвер і `nvidia-smi`;
- CUDA-сумісна версія OpenCV, зібрана з модулем `cudaarithm` (`WITH_CUDA=ON`);
- CMake та C++17-компілятор;

## Збірка й запуск

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/Users/32200/tools/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
$env:PATH = "$PWD\build\Release;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64;$env:PATH"
.\build\Release\cuda_video_difference.exe data\input\16756162_1280_720_25fps.mp4 frame_difference.mp4
```

Результат: `frame_difference.mp4`.
