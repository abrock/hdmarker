#include <stdio.h>

#include "hdmarker.hpp"
#include "timebench.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "ceres/ceres.h"

#include <iostream>

using namespace std;
using namespace cv;

const int grid_width = 16;
const int grid_height = 14;

const bool use_rgb = false;

const bool demosaic = false;

const float subfit_oversampling = 2.0;
const int subfit_max_size = 30;
const int subfit_min_size = 14;
const float min_fit_contrast = 3.0;

const float rms_use_limit = 2.0;

const double subfit_max_range = 0.5;

double max_accept_dist = 3.0;

#include <stdarg.h>

void printprogress(int curr, int max, int &last, const char *fmt = NULL, ...)
{
  last = (last + 1) % 4;
  int pos = curr*60/max;
  char unf[] = "                                                             ]";
  char fin[] = "[============================================================]";
  char buf[100];
  
  char cs[] = "-\\|/";
  memcpy(buf, fin, pos+1);
  buf[pos+1] = cs[last];
  memcpy(buf+pos+2, unf+pos+2, 62-pos-2+1);
  if (!fmt) {
    printf("%s\r", buf);
  }
  else {
    printf("%s", buf);
    va_list arglist;
    va_start(arglist, fmt);
    vprintf(fmt, arglist);
    va_end(arglist);
    printf("\r");
  }
  fflush(NULL);
}

void usage(const char *c)
{
  printf("usage: %s <input image> <output image>", c);
  exit(EXIT_FAILURE);
}

Point3f add_zero_z(Point2f p2)
{
  Point3f p = {p2.x, p2.y, 0.0};
  
  return p;
}

Point2f grid_to_world(Corner &c, int grid_w)
{
  Point2f p;
  int px, py;
  
  px = c.page % grid_w;
  py = c.page / grid_w;
  px*=32;
  py*=32;
  
  p.x = c.id.x+px;
  p.y = c.id.y+py;
  
  return p;
}

bool calib_savepoints(vector<vector<Point2f> > all_img_points[4], vector<vector<Point3f> > &all_world_points, vector<Corner> &corners, int hpages, int vpages, vector<Corner> &corners_filtered)
{
  if (!corners.size()) return false;
  
  vector<Point2f> plane_points;
  vector<Point2f> img_points[4];
  vector<Point3f> world_points;
  vector<Point2f> img_points_check[4];
  vector<Point2f> world_points_check;
  vector<int> pos;
  vector <uchar> inliers;
  
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
  Mat hom = findHomography(img_points_check[0], world_points_check, CV_RANSAC, 100, inliers);
  
  //vector<Point2f> proj;
  //perspectiveTransform(img_points_check[0], proj, hom);
  
  for(uint i=0;i<inliers.size();i++) {
    //printf("input %d distance %f\n", pos[i], norm(world_points_check[i]-proj[i]));
    if (!inliers[i])
      continue;
    corners_filtered.push_back(corners[pos[i]]);
    img_points[0].push_back((img_points_check[0])[i]);
    for(int c=1;c<4;c++)
      if (all_img_points[c].size())
        img_points[c].push_back((img_points_check[c])[i]);
    world_points.push_back(add_zero_z(world_points_check[i]));
  }
  
  printf("findHomography: %d inliers of %d calibration points (%.2f%%)\n", img_points[0].size(),img_points_check[0].size(),img_points[0].size()*100.0/img_points_check[0].size());

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
  rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  printf("rms %f with full distortion correction\n", rms);
    
  projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
  if (img.channels() == 1)
    cvtColor(img, paint, CV_GRAY2BGR);
  else
    paint = img.clone();
  resize(paint, paint, Size(Point2i(paint.size())*4), INTER_LINEAR);
  for(int i=0;i<projected.size();i++) {
    Point2f c = img_points[0][i]*4.0+Point2f(2,2);
    Point2f d = projected[i] - img_points[0][i];
    line(paint, c-Point2f(2,0), c+Point2f(2,0), Scalar(0,255,0));
    line(paint, c-Point2f(0,2), c+Point2f(0,2), Scalar(0,255,0));
    line(paint, c, c+10*d, Scalar(0,0,255));
  }
  
  imwrite("off_hdm.png", paint);
}

void check_calibration(vector<Corner> &corners, int w, int h, Mat &img, vector<Corner> &corners_filtered)
{
  vector<Mat> rvecs, tvecs;
  Mat cameraMatrix(3,3,cv::DataType<double>::type);
  Mat distCoeffs;
  double rms;
  vector<Point2f> projected;
  Mat paint;
  
  vector<vector<Point3f>> world_points(1);
  vector<vector<Point2f>> img_points[4];
  
  img_points[0].resize(1);
  if (use_rgb)
    for(int c=1;c<4;c++)
      img_points[c].resize(1);
    
  printf("corners: %d\n", corners.size());
    
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

/*
void check_precision(vector<Corner> &corners, int w, int h, Mat &img, const char *ref)
{
  vector<Mat> rvecs, tvecs;
  Mat cameraMatrix(3,3,cv::DataType<double>::type);
  Mat distCoeffs;
  double rms;
  vector<Point2f> projected;
  Mat paint;
  Mat refimg = imread(ref, CV_LOAD_IMAGE_GRAYSCALE);
  
  vector<vector<Point3f>> world_points(1);
  vector<vector<Point2f>> img_points(1);
  
  if (!calib_savepoints(img_points, world_points, corners, grid_width, grid_height)) {
    return;
  }
  
  distCoeffs = Mat::zeros(1, 8, CV_64F);
  rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  printf("rms %f with full distortion correction\n", rms);
    
  projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
  cvtColor(img, paint, CV_GRAY2BGR);
  for(int i=0;i<projected.size();i++) {
    Point2f d = projected[i] - img_points[0][i];
    line(paint, img_points[0][i], img_points[0][i]+100*d, Scalar(0,0,255));
  }
  imwrite("off_hdm.png", paint);
  
  cornerSubPix(refimg, img_points[0], Size(6,6), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 100, 0.001));
  
  distCoeffs = Mat::zeros(1, 8, CV_64F);
  rms = calibrateCamera(world_points, img_points, Size(w, h), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  printf("rms %f with full distortion correction, opencv cornerSubPix\n", rms);
  
  projectPoints(world_points[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, projected);
  cvtColor(img, paint, CV_GRAY2BGR);
  for(int i=0;i<projected.size();i++) {
    Point2f d = projected[i] - img_points[0][i];
    line(paint, img_points[0][i], img_points[0][i]+100*d, Scalar(0,0,255));
  }
  imwrite("off_ocv.png", paint);
}*/

bool corner_cmp(Corner a, Corner b)
{
  if (a.id.y != b.id.y)
    return a.id.y < b.id.y;
  else
    return a.id.x < b.id.x;
}

bool corner_find_off_save(vector<Corner> &corners, Corner ref, int x, int y, Point2f &out)
{
  std::vector<Corner>::iterator found;
  ref.id.x += x;
  ref.id.y += y;
  //FIXME doesn't work ?!
  /*bounds=equal_range(corners.begin(), corners.end(), ref, corner_cmp);
  if (bounds.first != bounds.second)
    return true;*/
  /*bool found = false;
  for(int i=0;i<corners.size();i++)
    if (corners[i].id == ref.id) {
      found = true;
      out = corners[i].p;
      break;
    }*/
  found = lower_bound(corners.begin(), corners.end(), ref, corner_cmp);
  if (found->id != ref.id)
    return true;
  out = found->p;
  return false;
}

struct Gauss2dError {
  Gauss2dError(int val, int x, int y, double m, double sw, double w)
      : val_(val), x_(x), y_(y), m_(m), sw_(sw), w_(w) {}

/**
 * used function: 
 */
  template <typename T>
  bool operator()(const T* const p,
                  T* residuals) const {
    T x2 = T(x_) - (T(m_)+sin(p[0])/T(subfit_oversampling)*T(subfit_max_range));
    T y2 = T(y_) - (T(m_)+sin(p[1])/T(subfit_oversampling)*T(subfit_max_range));
    T sx2 = T(2.0)*p[3]*p[3];
    T sy2 = T(2.0)*p[4]*p[4];
    x2 = x2*x2;
    y2 = y2*y2;
    
    //want to us sqrt(x2+y2)+T(1.0) but leads to invalid jakobian?
    //T d = sqrt(x2+y2+T(0.0001)) + T(w_);
    //T d = exp(-(x2/T(w_)+y2/T(w_)))+T(sw_);
    //non-weighted leads to better overall estimates?
    residuals[0] = (T(val_) - (p[5] + (p[2]-p[5])*exp(-(x2/sx2+y2/sy2))))*T(sw_);
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(int val, int x, int y, double m, double sw, double w) {
    return (new ceres::AutoDiffCostFunction<Gauss2dError, 1, 6>(
                new Gauss2dError(val, x, y, m, sw, w)));
  }

  int x_, y_, val_;
  double w_, sw_, m_;
};

/**
 * Fit 2d gaussian to image, 5 parameter: \f$x_0\f$, \f$y_0\f$, amplitude, spread, background
 * disregards a border of \f$\lfloor \mathit{size}/5 \rfloor\f$ pixels
 */
double fit_gauss(Mat &img, double *params)
{
  int size = img.size().width;
  assert(img.size().height == size);
  int b = size/5;
  uint8_t *ptr = img.ptr<uchar>(0);
  
  assert(img.depth() == CV_8U);
  assert(img.channels() == 1);
  
  double params_cpy[6];
  
  //x,y
  params[0] = size*0.5;
  params[1] = size*0.5;
  
  int sum = 0;
  int x, y = b;
  for(x=b;x<size-b-1;x++)
    sum += ptr[y*size+x];
  y = size-b-1;
  for(x=b+1;x<size-b;x++)
    sum += ptr[y*size+x];
  x = b;
  for(y=b+1;y<size-b;y++)
    sum += ptr[y*size+x];
  x = size-b-1;
  for(y=b;y<size-b-1;y++)
    sum += ptr[y*size+x];
  
  //background
  params[5] = sum / (4*(size-2*b-1));
  
  //amplitude
  params[2] = ptr[size/2*(size+1)];
  
  if (abs(params[2]-params[5]) < min_fit_contrast)
    return FLT_MAX;
  
  //spread
  params[3] = size*0.2;
  params[4] = size*0.2;
  
  for(int i=0;i<6;i++)
    params_cpy[i] = params[i];

  
  ceres::Problem problem_gauss;
  for(y=0;y<size-0;y++)
    for(x=0;x<size-0;x++) {
      double x2 = x-size*0.5;
      double y2 = y-size*0.5;
      x2 = x2*x2;
      y2 = y2*y2;
      double s2 = size*0.25;
      s2=s2*s2;
      double s = exp(-x2/s2-y2/s2);
      ceres::CostFunction* cost_function = Gauss2dError::Create(ptr[y*size+x], x, y, size*0.5, s, size*size*0.25);
      problem_gauss.AddResidualBlock(cost_function, NULL, params);
    }
  
  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::DENSE_QR;
  
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_gauss, &summary);
  
  params[0] = size*0.5+sin(params[0])/subfit_oversampling*subfit_max_range;
  params[1] = size*0.5+sin(params[1])/subfit_oversampling*subfit_max_range;
  
  Point2f v(params[0]-size*0.5, params[1]-size*0.5);
  
  if (summary.termination_type == ceres::CONVERGENCE)
    return sqrt(summary.final_cost/problem_gauss.NumResiduals());
  else
    return FLT_MAX;
}

void detect_sub_corners(Mat &img, vector<Corner> corners, vector<Corner> &corners_out, Mat &paint, int in_idx_step, float in_c_offset, float rms_use_limit, int out_idx_scale, int out_idx_offset)
{  
  int counter = 0;
  sort(corners.begin(), corners.end(), corner_cmp);
  
#pragma omp parallel for schedule(dynamic)
  for(int i=0;i<corners.size();i++) {
#pragma omp critical
    printprogress(i, corners.size(), counter, " %d subs", corners_out.size());
    vector<Point2f> ipoints(4);
    Corner c = corners[i];
    int size;
    
    ipoints[0] = corners[i].p;
    if (corner_find_off_save(corners, c, in_idx_step, 0, ipoints[1])) continue;
    if (corner_find_off_save(corners, c, in_idx_step, in_idx_step, ipoints[2])) continue;
    if (corner_find_off_save(corners, c, 0, in_idx_step, ipoints[3])) continue;
    
    float maxlen = 0;
    for(int i=0;i<4;i++) {
      Point2f v = ipoints[i]-ipoints[(i+1)%4];
      float len = v.x*v.x+v.y*v.y;
      if (len > maxlen)
        maxlen = len;
      v = ipoints[(i+3)%4]-ipoints[i];
      len = v.x*v.x+v.y*v.y;
      if (len > maxlen)
        maxlen = len;
    }
    maxlen = sqrt(maxlen);
    //printf("longest side %f\n", maxlen);
    
    if (maxlen < 5*4)
      continue;
    
    //FIXME need to use scale-space for perspective transform?
    size = std::max<int>(std::min<int>(maxlen*subfit_oversampling/5, subfit_max_size), subfit_min_size);
    
    for(int y=0;y<5;y++)
      for(int x=0;x<5;x++) {
        vector<Point2f> cpoints(4);
        cpoints[0] = Point2f((in_c_offset-x)*size,(in_c_offset-y)*size);
        cpoints[1] = Point2f((5+in_c_offset-x)*size,(in_c_offset-y)*size);
        cpoints[2] = Point2f((5+in_c_offset-x)*size,(5+in_c_offset-y)*size);
        cpoints[3] = Point2f((in_c_offset-x)*size,(5+in_c_offset-y)*size);
        
        Mat proj = Mat(size, size, CV_8U);
        Mat pers = getPerspectiveTransform(ipoints, cpoints);
        Mat pers_i = getPerspectiveTransform(cpoints, ipoints);
        warpPerspective(img, proj, pers, Size(size, size), INTER_LINEAR);
        
        double params[6];
        double rms = fit_gauss(proj, params);
        if (rms >= rms_use_limit)
          continue;
        
        vector<Point2f> coords(1);
        coords[0] = Point2f(params[0], params[1]);
        vector<Point2f> res(1);
        perspectiveTransform(coords, res, pers_i);
        
        Corner c_o(res[0], Point2i(c.id.x*out_idx_scale+2*x+out_idx_offset, c.id.y*out_idx_scale+2*y+out_idx_offset), 0);
#pragma omp critical
        corners_out.push_back(c_o);
      }

  }
  printf("\n");
}

void corrupt(Mat &img)
{
  GaussianBlur(img, img, Size(9,9), 0);
  Mat noise = Mat(img.size(), CV_32F);
  img. convertTo(img, CV_32F);
  randn(noise, 0, 3.0);
  img += noise;
  img.convertTo(img, CV_8U);
  cvtColor(img, img, COLOR_BayerRG2BGR_VNG);
  cvtColor(img, img, CV_BGR2GRAY);
}

int main(int argc, char* argv[])
{
  microbench_init();
  char buf[64];
  Mat img, paint;
  Point2f p1, p2;
  int x1, y1, x2, y2;
  vector<Corner> corners;
  Corner c;
  
  if (argc != 3 && argc != 4)
    usage(argv[0]);
  
  if (demosaic) {
    img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    paint = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cvtColor(img, img, COLOR_BayerRG2BGR);
    cvtColor(paint, paint, COLOR_BayerRG2BGR);
  }
  else {
    img = cv::imread(argv[1]);
    paint = cv::imread(argv[1]);
  }  
  
  //corrupt(img);
  //imwrite("corrupted.png", img);
  Marker::init();
  
  microbench_measure_output("app startup");
  //CALLGRIND_START_INSTRUMENTATION;
  if (argc == 4)
    Marker::detect(img, corners,use_rgb,0,0,atof(argv[3]),100);
  else
    Marker::detect(img, corners,use_rgb,0,0,0.5);
  //CALLGRIND_STOP_INSTRUMENTATION;
    
  microbench_init();
  
  printf("final score %d corners\n", corners.size());
  
  for(uint32_t i=0;i<corners.size();i++) {
    c = corners[i];
    //if (c.page != 256) continue;
    sprintf(buf, "%d/%d", c.id.x, c.id.y);
    circle(paint, c.p, 1, Scalar(0,0,0,0), 2);
    circle(paint, c.p, 1, Scalar(0,255,0,0));
    putText(paint, buf, c.p, FONT_HERSHEY_PLAIN, 0.5, Scalar(0,0,0,0), 2, CV_AA);
    putText(paint, buf, c.p, FONT_HERSHEY_PLAIN, 0.5, Scalar(255,255,255,0), 1, CV_AA);
  }
  
  vector<Corner> corners_f;
  check_calibration(corners, img.size().width, img.size().height, img, corners_f);
  //check_precision(corners, img.size().width, img.size().height, img, argv[3]);
  
  Mat gray;
  if (img.channels() != 1)
    cvtColor(img, gray, CV_BGR2GRAY);
  else
    gray = img;
  
  vector<Corner> corners_sub; 
  detect_sub_corners(gray , corners_f, corners_sub, paint, 1, 0.0, 100.0, 10, 1);
  
  vector<Corner> corners_f2;
  check_calibration(corners_sub, img.size().width, img.size().height, img, corners_f2);
  
  vector<Corner> corners_sub2; 
  detect_sub_corners(gray , corners_f2, corners_sub2, paint, 2, 0.5, rms_use_limit, 5, 0);
  
  vector<Corner> corners_f3;
  check_calibration(corners_sub2, img.size().width, img.size().height, img, corners_f3);
  
  imwrite(argv[2], paint);
  

  microbench_measure_output("app finish");
  return EXIT_SUCCESS;
}