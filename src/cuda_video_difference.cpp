#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/videoio.hpp>

#include <exception>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

struct FramePair {
    cv::Mat previous;
    cv::Mat current;
    cv::Mat difference;
};

void process_on_gpu(int device, int devices, std::vector<FramePair>& frames) {
    cv::cuda::setDevice(device);
    cv::cuda::GpuMat previous_gpu, current_gpu, difference_gpu;

    for (std::size_t i = device; i < frames.size(); i += devices) {
        previous_gpu.upload(frames[i].previous);
        current_gpu.upload(frames[i].current);
        cv::cuda::absdiff(current_gpu, previous_gpu, difference_gpu);
        difference_gpu.download(frames[i].difference);
    }
}

int run(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input.mp4 output.mp4\n";
        return 1;
    }

    const int devices = cv::cuda::getCudaEnabledDeviceCount();
    if (devices <= 0) {
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
    cv::VideoWriter output(argv[2], cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           input.get(cv::CAP_PROP_FPS), size);
    if (!output.isOpened()) {
        std::cerr << "Cannot create output video.\n";
        return 1;
    }

    output.write(cv::Mat::zeros(size, previous.type()));
    constexpr int pairs_per_gpu = 4;
    while (true) {
        std::vector<FramePair> frames;
        for (int i = 0; i < devices * pairs_per_gpu; ++i) {
            cv::Mat current;
            if (!input.read(current)) break;
            frames.push_back({previous, current, {}});
            previous = current;
        }
        if (frames.empty()) break;

        std::vector<std::thread> workers;
        for (int device = 0; device < devices; ++device) {
            workers.emplace_back(process_on_gpu, device, devices, std::ref(frames));
        }
        for (auto& worker : workers) worker.join();
        for (const auto& frame : frames) output.write(frame.difference);
    }

    std::cout << "Done with " << devices << " GPU(s): " << argv[2] << '\n';
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
