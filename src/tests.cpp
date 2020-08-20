#include <iostream>

#include <gtest/gtest.h>

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <algorithm>
#include <random>

#include "hdmarker.hpp"

std::random_device rd;
std::default_random_engine engine(rd());
std::normal_distribution<double> dist;

cv::Point2d randomPoint() {
    return cv::Point2d(dist(engine), dist(engine));
}

bool float_eq(float const a, float const b) {
    if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
        return false;
    }
    if (0 == a || 0 == b) {
        if (std::abs(a-b) < std::numeric_limits<float>::epsilon()) {
            return true;
        }
        else {
            return false;
        }
    }
    if (std::abs(a-b) / (std::abs(a) + std::abs(b)) < std::numeric_limits<float>::epsilon()) {
        return true;
    }
    return false;
}

bool point2f_eq(cv::Point2f const& a, cv::Point2f const& b) {
    return float_eq(a.x, b.x) && float_eq(a.y, b.y);
}

bool point2i_eq(cv::Point2i const& a, cv::Point2i const& b) {
    return a.x == b.x && a.y == b.y;
}

::testing::AssertionResult CornersEqual(hdmarker::Corner const& a, hdmarker::Corner const& b) {
    if (!point2f_eq(a.p, b.p)) {
        return ::testing::AssertionFailure() << "at p: " << a.p << " not equal to " << b.p;
    }
    for (size_t ii = 0; ii < 3; ++ii) {
        if (!point2f_eq(a.pc[ii], b.pc[ii])) {
            return ::testing::AssertionFailure() << "at pc[" << ii << "]: " << a.pc[ii] << " not equal to " << b.pc[ii];
        }
    }
    if (!point2i_eq(a.id, b.id)) {
        return ::testing::AssertionFailure() << "at id: " << a.id << " not equal to " << b.id;
    }
    if (a.page != b.page) {
        return ::testing::AssertionFailure() << "at page: " << a.page << " not equal to " << b.page;
    }
    if (!float_eq(a.size, b.size)) {
        return ::testing::AssertionFailure() << "at size: " << a.size << "not equal to " << b.size;
    }
    if (a.level != b.level) {
        return ::testing::AssertionFailure() << "at level: " << a.level << " not equal to " << b.level;
    }
    if (a.color != b.color) {
        return ::testing::AssertionFailure() << "at page: " << a.color << " not equal to " << b.color;
    }
    return ::testing::AssertionSuccess();
}

TEST(Corner, opencv_storage) {
    hdmarker::Corner a, b;
    a.p = cv::Point2f(1,2);
    a.id = cv::Point2i(3,4);
    a.pc[0] = cv::Point2f(5,6);
    a.pc[1] = cv::Point2f(7,8);
    a.pc[2] = cv::Point2f(9,10);
    a.page = 11;
    a.size = 12;
    a.level = 13;
    a.color = 14;
    std::string const storage_file = "asdfghjk-test-temp-storage.yaml";

    {
        cv::FileStorage pointcache(storage_file, cv::FileStorage::WRITE);
        pointcache << "a" << a;
        pointcache.release();
    }
    {
        cv::FileStorage pointcache(storage_file, cv::FileStorage::READ);
        pointcache["a"] >> b;
    }
    EXPECT_TRUE(CornersEqual(a,b));
}

TEST(Corner, opencv_storage_vec) {
    std::vector<hdmarker::Corner> src(10*1000);
    std::vector<hdmarker::Corner> dst;

    for (size_t ii = 0; ii < src.size(); ++ii) {
        src[ii].p.x = dist(engine);
        src[ii].p.y = dist(engine);
        src[ii].pc[0] = {dist(engine), dist(engine)};
        src[ii].pc[1] = {dist(engine), dist(engine)};
        src[ii].pc[2] = {dist(engine), dist(engine)};
        src[ii].page = dist(engine);
        src[ii].size = dist(engine);
        src[ii].level = dist(engine);
        src[ii].color = dist(engine);
    }

    std::string const storage_file = "asdfghjk-test-temp-storage-vector.yaml.gz";

    {
        cv::FileStorage pointcache(storage_file, cv::FileStorage::WRITE);
        pointcache << "vec" << src;
        pointcache.release();
    }
    {
        cv::FileStorage pointcache(storage_file, cv::FileStorage::READ);
        pointcache["vec"] >> dst;
    }
    EXPECT_EQ(src.size(), dst.size());
    for (size_t ii = 0; ii < src.size() && ii < dst.size(); ++ii) {
        EXPECT_TRUE(CornersEqual(src[ii], dst[ii]));
    }
}

TEST(Corner, binary_storage_vec) {
    std::vector<hdmarker::Corner> src(10*1000);
    std::vector<hdmarker::Corner> dst, dst_bin;

    for (size_t ii = 0; ii < src.size(); ++ii) {
        src[ii].p.x = dist(engine);
        src[ii].p.y = dist(engine);
        src[ii].pc[0] = {dist(engine), dist(engine)};
        src[ii].pc[1] = {dist(engine), dist(engine)};
        src[ii].pc[2] = {dist(engine), dist(engine)};
        src[ii].page = dist(engine);
        src[ii].size = dist(engine);
        src[ii].level = dist(engine);
        src[ii].color = dist(engine);
    }

    std::string const storage_file = "asdfghjk-test-temp-storage-vector.hdmarker";

    std::stringstream out;
    hdmarker::Corner::writeStream(out, src);
    std::stringstream in(out.str());
    hdmarker::Corner::readStream(in, dst);

    EXPECT_EQ(src.size(), dst.size());
    for (size_t ii = 0; ii < src.size() && ii < dst.size(); ++ii) {
        EXPECT_TRUE(CornersEqual(src[ii], dst[ii]));
    }

    dst.clear();
    hdmarker::Corner::writeFile(storage_file, src);
    hdmarker::Corner::readFile(storage_file, dst);
    EXPECT_EQ(src.size(), dst.size());
    for (size_t ii = 0; ii < src.size() && ii < dst.size(); ++ii) {
        EXPECT_TRUE(CornersEqual(src[ii], dst[ii]));
    }
}

TEST(Corner, gzipped_binary_storage_vec) {
    std::vector<hdmarker::Corner> src(10*1000);
    std::vector<hdmarker::Corner> dst, dst_bin;

    for (size_t ii = 0; ii < src.size(); ++ii) {
        src[ii].p.x = dist(engine);
        src[ii].p.y = dist(engine);
        src[ii].pc[0] = {dist(engine), dist(engine)};
        src[ii].pc[1] = {dist(engine), dist(engine)};
        src[ii].pc[2] = {dist(engine), dist(engine)};
        src[ii].page = dist(engine);
        src[ii].size = dist(engine);
        src[ii].level = dist(engine);
        src[ii].color = dist(engine);
    }

    std::string const storage_file = "asdfghjk-test-temp-storage-vector.hdmarker.gz";

    dst.clear();
    hdmarker::Corner::writeFile(storage_file, src);
    hdmarker::Corner::readFile(storage_file, dst);
    EXPECT_EQ(src.size(), dst.size());
    for (size_t ii = 0; ii < src.size() && ii < dst.size(); ++ii) {
        EXPECT_TRUE(CornersEqual(src[ii], dst[ii]));
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    std::cout << "RUN_ALL_TESTS return value: " << RUN_ALL_TESTS() << std::endl;

}
