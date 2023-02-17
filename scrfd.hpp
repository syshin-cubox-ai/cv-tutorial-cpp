#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "NumCpp.hpp"

void resize_preserving_aspect_ratio(cv::Mat &img, float &scale, int img_size, float scale_ratio=1.0f);

void transform_image(cv::Mat &img, float &scale, float scale_ratio=1.0f);

int non_max_suppression(nc::NdArray<float> &pred, float conf_thres, float iou_thres);

void clip_coords(nc::NdArray<float> &pred, nc::Shape img_shape);

void parse_prediction(nc::NdArray<float> pred, nc::NdArray<int> &bbox, nc::NdArray<float> &conf, nc::NdArray<int> &kps);
