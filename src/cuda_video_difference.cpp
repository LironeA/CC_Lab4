#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "frame_difference.cuh"

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
    for (std::size_t i = device; i < frames.size(); i += devices) {
        frames[i].difference.create(frames[i].current.size(), frames[i].current.type());
        difference_on_gpu(device, frames[i].previous.data, frames[i].current.data,
                          frames[i].difference.data, frames[i].current.total() * frames[i].current.elemSize());
    }
}

int run(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input.mp4 output.mp4\n";
        return 1;
    }

    const int devices = available_gpu_count();
    if (devices == 0) {
        std::cerr << "An NVIDIA GPU is required.\n";
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
    if (!previous.isContinuous()) previous = previous.clone();

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
            if (!current.isContinuous()) current = current.clone();
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
