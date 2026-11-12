# Лабораторна №4 — CUDA-обробка відео

Програма читає MP4 через OpenCV, вручну передає послідовні кадри в пам'ять усіх доступних NVIDIA GPU через CUDA Runtime API, обчислює їх абсолютну попарну різницю власним CUDA kernel та записує MP4-результат через OpenCV. Нульовий кадр результату чорний, адже для нього немає попереднього кадру. CPU fallback навмисно відсутній.

Кадри обробляються пакетами. Для кожного GPU створюється окремий C++ потік; він отримує свою частину пар кадрів і виконує `cudaMalloc`, `cudaMemcpy`, власний kernel та `cudaFree` на призначеному пристрої. Після завершення пакета результати записуються у початковому порядку кадрів.

## Вхідні дані

`data/input/16756162_1280_720_25fps.mp4` — наданий ролик 1280×720, 25 FPS.

## Вимоги

- NVIDIA GPU, драйвер і `nvidia-smi`;
- CUDA Toolkit;
- OpenCV C++ із модулями `core` і `videoio`;
- CMake та C++17-компілятор;

## Збірка й запуск

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/Users/32200/tools/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
$env:PATH = "$PWD\build\Release;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64;$env:PATH"
.\build\Release\cuda_video_difference.exe data\input\16756162_1280_720_25fps.mp4 frame_difference.mp4
```

Результат: `frame_difference.mp4`.
