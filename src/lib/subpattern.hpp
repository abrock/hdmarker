#ifndef HDMARKER_SUBPATTERN_H
#define HDMARKER_SUBPATTERN_H

#include <opencv2/core/core.hpp>

#include "hdmarker.hpp"

namespace hdmarker {
  
  
void hdmarker_subpattern_step(cv::Mat &img, std::vector<Corner> corners, std::vector<Corner> &corners_out, int in_idx_step, float in_c_offset, int out_idx_scale, int out_idx_offset, bool ignore_corner_neighbours = false, cv::Mat *paint = NULL, bool *mask_2x2 = NULL);
enum flag {KEEP_ALL_LEVELS = 1, KEEP_ALL_RECURSIVE_LEVELS = 2};

void hdmarker_detect_subpattern(cv::Mat &img, std::vector<Corner> corners, std::vector<Corner> &corners_out, int depth, double *size, cv::Mat *paint = NULL, bool *mask_2x2 = NULL, int page = -1, const cv::Rect limit = cv::Rect(), int flags = KEEP_ALL_LEVELS);

}

#endif
