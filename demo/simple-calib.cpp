/**
* @file simple-calib.cpp
* @brief demo calibration application of hdmarker
*
* @author Alexander Brock
* @date 02/20/2019
*/

#include <stdio.h>
#include <map>
#include <iostream>
#include <unordered_map>

#include "hdmarker.hpp"
//#include "timebench.hpp"
#include "subpattern.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <tclap/CmdLine.h>

#include <boost/filesystem.hpp>

const int grid_width = 1;
const int grid_height = 1;

const bool use_rgb = false;

using namespace std;
using namespace hdmarker;
using namespace cv;

namespace fs = boost::filesystem;

cv::Point3f add_zero_z(cv::Point2f p2)
{
    cv::Point3f p = {p2.x, p2.y, 0.0};

    return p;
}

cv::Point2f grid_to_world(hdmarker::Corner &c, int grid_w)
{
    cv::Point2f p;
    int px, py;

    px = c.page % grid_w;
    py = c.page / grid_w;
    px*=32;
    py*=32;

    p.x = c.id.x+px;
    p.y = c.id.y+py;

    return p;
}

bool calib_savepoints(
        std::vector<std::vector<cv::Point2f> > all_img_points[4],
        std::vector<std::vector<cv::Point3f> > &all_world_points,
        std::vector<hdmarker::Corner> &corners,
        int hpages, int vpages,
        std::vector<hdmarker::Corner> &corners_filtered)
{
    if (!corners.size()) return false;

    std::vector<cv::Point2f> plane_points;
    std::vector<cv::Point2f> img_points[4];
    std::vector<cv::Point3f> world_points;
    std::vector<cv::Point2f> img_points_check[4];
    std::vector<cv::Point2f> world_points_check;
    std::vector<int> pos;
    std::vector <uchar> inliers;

    for(uint i=0;i<corners.size();i++) {
        int px, py;
        if (corners[i].page > hpages*vpages)
            continue;
        /*if (corners[i].page != 0)
      continue;*/
        px = corners[i].page % hpages;
        py = corners[i].page / hpages;
        px*=32;
        py*=32;
        img_points_check[0].push_back(corners[i].p);
        pos.push_back(i);
        for(int c=1;c<4;c++)
            if (all_img_points[c].size())
                img_points_check[c].push_back(corners[i].pc[c-1]);
        world_points_check.push_back(grid_to_world(corners[i], hpages));
    }

    inliers.resize(world_points_check.size());
    cv::Mat hom = findHomography(world_points_check, img_points_check[0], cv::RANSAC, 100, inliers);

    std::vector<cv::Point2f> proj;
    perspectiveTransform(world_points_check, proj, hom);

    double rms = 0.0;
    for(uint i=0;i<inliers.size();i++) {
        if (!inliers[i])
            continue;
        rms += norm(img_points_check[0][i]-proj[i])*norm(img_points_check[0][i]-proj[i]);
        corners_filtered.push_back(corners[pos[i]]);
        img_points[0].push_back((img_points_check[0])[i]);
        for(int c=1;c<4;c++)
            if (all_img_points[c].size())
                img_points[c].push_back((img_points_check[c])[i]);
        world_points.push_back(add_zero_z(world_points_check[i]));
    }
    printf("homography rms: %f\n", sqrt(rms/inliers.size()));

    printf("findHomography: %lu inliers of %lu calibration points (%.2f%%)\n", img_points[0].size(),img_points_check[0].size(),img_points[0].size()*100.0/img_points_check[0].size());

    (all_img_points[0])[0] = img_points[0];
    for(int c=1;c<4;c++)
        if (all_img_points[c].size())
            (all_img_points[c])[0] = img_points[c];
    all_world_points[0] = world_points;

    return true;
}

void calibrate_channel(vector<vector<Point2f> > &img_points, vector<vector<Point3f> > &world_points, int w, int h, Mat &img)
{
    vector<Mat> rvecs, tvecs;
    Mat cameraMatrix(3,3,cv::DataType<double>::type);
    Mat distCoeffs;
    double rms;
    vector<Point2f> projected;
    Mat paint;

    distCoeffs = Mat::zeros(1, 8, CV_64F);
    //use CV_CALIB_ZERO_TANGENT_DIST or we get problems when using single image!
    rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_RATIONAL_MODEL);
    printf("rms %f with full distortion correction\n", rms);

    cout << distCoeffs << endl;

    projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
    if (img.channels() == 1)
        cvtColor(img, paint, cv::COLOR_GRAY2BGR);
    else
        paint = img.clone();
    //resize(paint, paint, Size(Point2i(paint.size())*4), INTER_LINEAR);
    for(uint i=0;i<projected.size();i++) {
        /*Point2f c = img_points[0][i]*4.0+Point2f(2,2);
    Point2f d = projected[i] - img_points[0][i];
    line(paint, c-Point2f(2,0), c+Point2f(2,0), Scalar(0,255,0));
    line(paint, c-Point2f(0,2), c+Point2f(0,2), Scalar(0,255,0));
    line(paint, c, c+10*d, Scalar(0,0,255));*/

        Point2f c = img_points[0][i];
        Point2f d = img_points[0][i]-projected[i];
        line(paint, c-Point2f(2,0), c+Point2f(2,0), Scalar(0,255,0));
        line(paint, c-Point2f(0,2), c+Point2f(0,2), Scalar(0,255,0));
        line(paint, c, c+10*d, Scalar(0,0,255));
    }

    imwrite("off_hdm.jpg", paint);
}

void check_calibration(vector<Corner> &corners, int w, int h, Mat &img, vector<Corner> &corners_filtered)
{
    vector<Mat> rvecs, tvecs;
    Mat cameraMatrix(3,3,cv::DataType<double>::type);
    Mat distCoeffs;
    vector<Point2f> projected;
    Mat paint;

    vector<vector<Point3f>> world_points(1);
    vector<vector<Point2f>> img_points[4];

    img_points[0].resize(1);
    if (use_rgb)
        for(int c=1;c<4;c++)
            img_points[c].resize(1);

    printf("corners: %lu\n", corners.size());

    if (!calib_savepoints(img_points, world_points, corners, grid_width, grid_height, corners_filtered)) {
        return;
    }

    calibrate_channel(img_points[0], world_points, w, h, img);
    if (use_rgb)
        for(int c=1;c<4;c++)
            calibrate_channel(img_points[c], world_points, w, h, img);


    /*cornerSubPix(img, img_points[0], Size(4,4), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 100, 0.001));

  distCoeffs = Mat::zeros(1, 8, CV_64F);
  rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  printf("rms %f with full distortion correction, opencv cornerSubPix\n", rms);

  projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
  cvtColor(img, paint, CV_GRAY2BGR);
  for(int i=0;i<projected.size();i++) {
    Point2f d = projected[i] - img_points[0][i];
    line(paint, img_points[0][i], img_points[0][i]+100*d, Scalar(0,0,255));
  }
  imwrite("off_ocv.png", paint);*/
}

void corrupt(Mat &img)
{
    GaussianBlur(img, img, Size(9,9), 0);
    Mat noise = Mat(img.size(), CV_32F);
    img. convertTo(img, CV_32F);
    randn(noise, 0, 3.0);
    img += noise;
    img.convertTo(img, CV_8U);
    cvtColor(img, img, COLOR_BayerBG2BGR_VNG);
    cvtColor(img, img, cv::COLOR_BGR2GRAY);
}

vector<Corner> getCorners(
        const std::string input_file,
        const float effort,
        const bool demosaic,
        const int recursion_depth
        ) {
    std::string pointcache_file = input_file + "-pointcache.yaml";
    vector<Corner> corners;
    bool read_cache_success = false;



    try {
        if (fs::exists(pointcache_file)) {
            read_cache_success = true;
            FileStorage pointcache(pointcache_file, FileStorage::READ);
            FileNode n = pointcache["corners"]; // Read string sequence - Get node
            if (n.type() != FileNode::SEQ)
            {
                cerr << "corners is not a sequence! FAIL" << endl;
                read_cache_success = false;
            }

            FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
            for (; it != it_end; ++it) {
                hdmarker::Corner c;
                *it >> c;
                corners.push_back(c);
            }

        }
    }
    catch (const Exception& e) {
        std::cout << "Reading pointcache file failed with exception: " << std::endl
                  << e.what() << std::endl;
        read_cache_success = false;
    }

    if (read_cache_success) {
        std::cout << "Got corners from pointcache file" << std::endl;
        return corners;
    }




    //  microbench_init();
    char buf[64];
    Mat img, paint;
    Point2f p1, p2;
    int x1, y1, x2, y2;

    Corner c;

    if (demosaic) {
        img = cv::imread(input_file, cv::IMREAD_GRAYSCALE);
        cvtColor(img, img, COLOR_BayerBG2BGR);
        paint = img.clone();
    }
    else {
        img = cv::imread(input_file);
        paint = img.clone();
    }
    std::cout << "Input image size: " << img.size << std::endl;

    //corrupt(img);
    //imwrite("corrupted.png", img);
    Marker::init();

    //  microbench_measure_output("app startup");

    //CALLGRIND_START_INSTRUMENTATION;
    detect(img, corners,use_rgb,0,10, effort, 3);
    //CALLGRIND_STOP_INSTRUMENTATION;

    //  microbench_init();

    printf("final score %zu corners\n", corners.size());

    for(size_t i=0;i<corners.size();i++) {
        c = corners[i];
        //if (c.page != 256) continue;
        sprintf(buf, "%d/%d", c.id.x, c.id.y);
        circle(paint, c.p, 1, Scalar(0,0,0,0), 2);
        circle(paint, c.p, 1, Scalar(0,255,0,0));
        putText(paint, buf, c.p, FONT_HERSHEY_PLAIN, 0.5, Scalar(0,0,0,0), 2, cv::LINE_AA);
        putText(paint, buf, c.p, FONT_HERSHEY_PLAIN, 0.5, Scalar(255,255,255,0), 1, cv::LINE_AA);
    }
    imwrite(input_file + "-1.png", paint);

    //
    Mat gray;
    if (img.channels() != 1)
        cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;

    if (recursion_depth > 0) {
        std::cout << "Drawing sub-markers" << std::endl;
        vector<Corner> corners_sub;
        double msize = 1.0;
        refine_recursive(gray, corners, corners_sub, 3, &msize);

        vector<Corner> corners_f2;
        check_calibration(corners_sub, img.size().width, img.size().height, img, corners_f2);

        cv::Mat paint2 = paint.clone();
        for (hdmarker::Corner const& c : corners_f2) {
            circle(paint2, c.p, 3, Scalar(0,0,255,0), -1, LINE_AA);
        }
        imwrite(input_file + "-2.png", paint2);
    }

    FileStorage pointcache(pointcache_file, FileStorage::WRITE);
    pointcache << "corners" << "[";
    for (hdmarker::Corner const& c : corners) {
        pointcache << c;
    }
    pointcache << "]";

    return corners;

}

int main(int argc, char* argv[])
{

    FileStorage pointcache("testcache", FileStorage::WRITE);

    hdmarker::Corner p(Point(1,2), Point(3,4), 5);
    pointcache << "corner" << "[" << p << "]";

    pointcache.release();

    pointcache.open("testcache", FileStorage::READ);

    hdmarker::Corner q;

    FileNode fn = pointcache["corner"];
    int id=0;
    for (FileNodeIterator it = fn.begin(); it != fn.end(); it++,id++) {
        *it >> q;
        std::cout << q.p;
    }


    std::vector<std::string> input_files;
    int recursion_depth = -1;
    float effort = 0.5;
    bool demosaic = false;
    try {
        TCLAP::CmdLine cmd("hdmarker calibration tool", ' ', "0.1");

        TCLAP::ValueArg<int> recursive_depth_arg("r", "recursion",
                                                 "Recursion depth of the sub-marker detection. Set this to the actual recursion depth of the used target.",
                                                 false, -1, "int");
        cmd.add(recursive_depth_arg);

        TCLAP::ValueArg<float> effort_arg("e", "effort",
                                          "Effort value for the marker detection.",
                                          false, .5, "float");
        cmd.add(effort_arg);

        TCLAP::SwitchArg demosaic_arg("d", "demosaic",
                                      "Use this flag if the input images are raw images and demosaicing should be used.",
                                      false);

        TCLAP::UnlabeledMultiArg<string> input_img_arg("input", "Input images, should contain markers", true, "string");
        cmd.add(input_img_arg);


        cmd.parse(argc, argv);


        input_files = input_img_arg.getValue();
        recursion_depth = recursive_depth_arg.getValue();
        effort = effort_arg.getValue();
        demosaic = demosaic_arg.getValue();

        std::cout << "Parameters: " << std::endl
                  << "Number of input files: " << input_files.size() << std::endl
                  << "recursion depth: " << recursion_depth << std::endl
                  << "effort: " << effort << std::endl
                  << "demosaic: " << (demosaic ? "true" : "false") << std::endl;
    }
    catch (TCLAP::ArgException const & e) {
        std::cerr << e.what() << std::endl;
        return 0;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 0;
    }

    std::map<std::string, std::vector<hdmarker::Corner> > detected_markers;

    for (std::string const& input_file : input_files) {
        detected_markers[input_file] = getCorners(input_file, effort, demosaic, recursion_depth);
    }


    //  microbench_measure_output("app finish");
    return EXIT_SUCCESS;
}
