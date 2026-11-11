#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <exception>
#include <iostream>

int run(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input.mp4 output.mp4\n";
        return 1;
    }
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
        std::cerr << "CUDA-enabled OpenCV and an NVIDIA GPU are required.\n";
        return 1;
    }

    cv::VideoCapture input(argv[1]);
    if (!input.isOpened()) {
        std::cerr << "Cannot open input video.\n";
        return 1;
    }

    cv::Mat previous;
    if (!input.read(previous)) {
        std::cerr << "Input video has no frames.\n";
        return 1;
    }

    const cv::Size size = previous.size();
    const double fps = input.get(cv::CAP_PROP_FPS);
    std::filesystem::path output_path(argv[2]);
    if (output_path.has_parent_path()) std::filesystem::create_directories(output_path.parent_path());
    cv::VideoWriter output(argv[2], cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
    if (!output.isOpened()) {
        std::cerr << "Cannot create output video.\n";
        return 1;
    }

    cv::cuda::GpuMat previous_gpu, current_gpu, difference_gpu;
    previous_gpu.upload(previous);
    output.write(cv::Mat::zeros(size, previous.type()));

    cv::Mat current, difference;
    while (input.read(current)) {
        current_gpu.upload(current);
        cv::cuda::absdiff(current_gpu, previous_gpu, difference_gpu);
        difference_gpu.download(difference);
        output.write(difference);
        current_gpu.copyTo(previous_gpu);
    }

    std::cout << "Done: " << argv[2] << '\n';
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const cv::Exception& error) {
        std::cerr << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }
    return 1;
}
