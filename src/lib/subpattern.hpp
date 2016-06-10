#ifndef HDMARKER_SUBPATTERN_H
#define HDMARKER_SUBPATTERN_H

#include <opencv2/core/core.hpp>

#include "hdmarker.hpp"

namespace hdmarker {
  
enum flag {KEEP_ALL_LEVELS = 1, KEEP_ALL_RECURSIVE_LEVELS = 2};

void hdmarker_detect_subpattern(cv::Mat &img, std::vector<Corner> corners, std::vector<Corner> &corners_out, int depth, double *size, cv::Mat *paint, bool *mask_2x2, int page, const cv::Rect &limit, int flags = KEEP_ALL_LEVELS);

void hdmarker_detect_subpattern(cv::Mat &img, std::vector<Corner> corners, std::vector<Corner> &corners_out, int depth, double *size, cv::Mat *paint = NULL, bool *mask_2x2 = NULL, int page = -1, const std::vector<cv::Rect> &limits = std::vector<cv::Rect>(), int flags = KEEP_ALL_LEVELS);

void hdmarker_subpattern_set_gt_mats(cv::Mat &c, cv::Mat &r, cv::Mat &t);

}

#endif
