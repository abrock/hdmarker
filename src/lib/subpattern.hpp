#ifndef HDMARKER_SUBPATTERN_H
#define HDMARKER_SUBPATTERN_H

/** 
* @file subpattern.hpp 
* @brief detect fractal points recursively
*
* @author Hendrik Schilling (implementation)
* @author Maximilian Diebold (documentation)
* @date 01/15/2018
*/

#include <opencv2/core/core.hpp>
#include "hdmarker.hpp"

#include <exception>

namespace hdmarker {
  
class runaway_subpattern : public std::exception
{
public:
  std::string err;
  runaway_subpattern(std::string err) : err(err) {};
};
  
enum flag {KEEP_ALL_LEVELS = 1, KEEP_ALL_RECURSIVE_LEVELS = 2};

void refine_recursive(cv::Mat &img, 
                      std::vector<Corner> corners, 
                      std::vector<Corner> &corners_out, 
                      int depth, 
                      double *size, 
                      cv::Mat *paint, 
                      bool *mask_2x2, 
                      int page, 
                      const cv::Rect &limit, 
                      int flags = KEEP_ALL_LEVELS);

/** Recursively detect and the calibration points of the fractal calibration pattern.
 @param[in] img takes an input image
 \param[in] corners input corners as generated by \a hdmarker::detect()
 \param[out] corners_out detected calibration points from all scales
 \param[in,out] size marker size
 \param[in] depth maximum recursion depth
 \param[out] paint debug output - show detected calibration points
 \param[in] mask_2x2 boolean mask indicating which pixels on an 2x2 grid are valid for refinement (useful for detection with bayer sensors
 \param[in] page valid page for input corners
 \param[in] limit list of valid target locations (in marker coordinates)
 \param[in] flags indicate which corners to output to corners_out - normally all corners from all detected scales
 
 Because the location accuracy on the calibration target is an integer coordiante, with a resolution increased five times for every recursion step, the \a size parameters allows to keep track of the scale of the corners. size should be set to the real-world (measured) marker size and after a call to refine_recursive() will contain a new value which, when multiplied with the corner id gives the real-world position on the calibration target:

\code{.cpp}
//marker size is 15.7mm
double multiplier = 15.7;

refine_recursive(img, corners_rough, corners, 3, &multiplier);

for(int i=0;i<corners.size();i++) {
  ipoints[i] = corners[i].p;
  //target coordinates in mm (to scale!)
  wpoints[i] = Point3f(multiplier*corners[i].id.x, multiplier*corners[i].id.y, 0);
}
\endcode
The \a paint parameter optionally points to a matrix which will be used to draw all detected calibration points, useful for visualization. The 2x2 mask contains a boolean mask which describes which pixels should be used (true) and which should be ignored (false) - this can be used for direct bayer pattern refinement.

The \a page parameter allows to select which page id is valid for refinement - This can improve the robustness, but is not generally needed.

*/
void refine_recursive(cv::Mat &img, 
                      std::vector<Corner> corners, 
                      std::vector<Corner> &corners_out, 
                      int depth, double *size, 
                      cv::Mat *paint = NULL, 
                      bool *mask_2x2 = NULL, 
                      int page = -1, 
                      const std::vector<cv::Rect> &limits = std::vector<cv::Rect>(), 
                      int flags = KEEP_ALL_LEVELS);


void hdmarker_subpattern_set_gt_mats(cv::Mat &c, cv::Mat &r, cv::Mat &t);

}

#endif
