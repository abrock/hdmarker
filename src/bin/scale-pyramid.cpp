#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

boost::system::error_code ignore_error;

void create_pyramid(std::string const& _src) {
    cv::Mat_<uint16_t> const src = cv::imread(_src, cv::IMREAD_GRAYSCALE)*256;

    fs::path dst_dir = _src + "-pyramid";
    fs::create_directories(dst_dir, ignore_error);

    cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f(src.size().width/2, src.size().height/2), 1, 1.0);

    double const blur = 1;
    double const pre_blur = .7;
    size_t counter = 0;
    for (double scale = .3*11/15; scale >= 0.07; scale *= .95) {
        cv::Mat_<uint16_t> dst = src.clone();
        dst *= 256;
        dst /= 1.5;
        dst += 42;
        cv::Mat_<uint16_t> noise(dst.size(), uint16_t(0));
        cv::randn(noise, 0, 2*256);
        dst += noise;
        double const local_blur = blur * scale; //std::max(1.0, blur*scale);
        cv::GaussianBlur(dst, dst, cv::Size(), pre_blur/scale, pre_blur/scale);
        cv::warpAffine(dst, dst, rotation, dst.size(), cv::INTER_CUBIC, cv::BORDER_REFLECT);
        cv::resize(dst, dst, cv::Size(), scale, scale, cv::INTER_CUBIC);
        cv::GaussianBlur(dst, dst, cv::Size(), local_blur, local_blur);

        cv::Mat_<uint16_t> noise2(dst.size(), uint16_t(0));
        cv::randn(noise2, 0, 2*256);
        dst += noise2;

        double minval = 255;
        double maxval = 0;
        cv::minMaxIdx(dst, & minval, & maxval);

        //dst -= minval;
        //dst *= 255.0/(maxval - minval);
        //dst -= 63;
        //dst *= 2;
        std::cout << "min/max: " << minval << " / " << maxval << std::endl;

        cv::imwrite((dst_dir / (std::to_string(counter) + ".png")).string(), dst);
        counter++;
    }
}

int main(int argc, char ** argv) {

    for (size_t ii = 1; ii < argc; ++ii) {
        create_pyramid(argv[ii]);
    }
}
