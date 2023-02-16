#include <iostream>
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "NumCpp.hpp"

using namespace std;

void resize_preserving_aspect_ratio(cv::Mat &img, float &scale, int img_size, float scale_ratio=1.0f)
{
    int h = img.rows;
    int w = img.cols;
    scale = round(img_size / scale_ratio) / max(h, w);
    if (scale != 1.0f)
    {
        cv::InterpolationFlags interpolation = (scale < 1) ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(img, img, cv::Size(), scale, scale, interpolation);
    }
}

void transform_image(cv::Mat &img, float &scale, float scale_ratio=1.0f)
{
    int img_size = 640;
    resize_preserving_aspect_ratio(img, scale, img_size);

    int pad[] = {0, img_size - img.rows, 0, img_size - img.cols};
    cv::copyMakeBorder(img, img, pad[0], pad[1], pad[2], pad[3], cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    img.convertTo(img, CV_32FC3);
}

int non_max_suppression(nc::NdArray<float> &pred, float conf_thres, float iou_thres)
{
    // 임계값 이상으로 confidence를 가진 항목만 남김
    nc::NdArray<nc::uint32> keep = nc::flatnonzero(pred(pred.rSlice(), 4) >= conf_thres);
    if (keep.size() == 0)
        return -1;
    pred = pred(keep, pred.cSlice());

    // 속도 개선을 위해, 추후 IoU 계산에 사용할 각 bbox의 넓이를 한 번에 계산해둠
    nc::NdArray<float> x1 = nc::transpose(pred(pred.rSlice(), 0));
    nc::NdArray<float> y1 = nc::transpose(pred(pred.rSlice(), 1));
    nc::NdArray<float> x2 = nc::transpose(pred(pred.rSlice(), 2));
    nc::NdArray<float> y2 = nc::transpose(pred(pred.rSlice(), 3));
    nc::NdArray<float> score = nc::transpose(pred(pred.rSlice(), 4));
    nc::NdArray<float> area = (x2 - x1) * (y2 - y1);

    nc::NdArray<nc::uint32> order = nc::flip(score.argsort());
    vector<uint32_t> output_idx;
    while (order.size() > 0)
    {
        // confidence가 가장 높은 항목(order(0, 0))을 선택하고, 최종 출력 리스트에 추가
        nc::NdArray<nc::uint32> highest_conf_item = {order(0, 0)};
        output_idx.push_back(highest_conf_item.item());

        // order(0, 0)을 기준으로 나머지 모든 항목에 대해 IoU를 구함
        nc::NdArray<nc::uint32> rest_item = order(0, order.cSlice(1));
        nc::NdArray<float> inter_x1 = nc::maximum(nc::repeat(x1(0, highest_conf_item), 1U, rest_item.shape().cols), x1(0, rest_item));
        nc::NdArray<float> inter_y1 = nc::maximum(nc::repeat(y1(0, highest_conf_item), 1U, rest_item.shape().cols), y1(0, rest_item));
        nc::NdArray<float> inter_x2 = nc::minimum(nc::repeat(x2(0, highest_conf_item), 1U, rest_item.shape().cols), x2(0, rest_item));
        nc::NdArray<float> inter_y2 = nc::minimum(nc::repeat(y2(0, highest_conf_item), 1U, rest_item.shape().cols), y2(0, rest_item));
        nc::NdArray<float> w = nc::maximum(nc::zeros<float>(1, rest_item.shape().cols), inter_x2 - inter_x1);
        nc::NdArray<float> h = nc::maximum(nc::zeros<float>(1, rest_item.shape().cols), inter_y2 - inter_y1);
        nc::NdArray<float> intersection = w * h;
        nc::NdArray<float> o_union = nc::repeat(area(0, highest_conf_item), 1U, rest_item.shape().cols) + area(0, rest_item) - intersection;
        nc::NdArray<float> iou = intersection / o_union;

        // 임계값 이하로 iou를 가진 항목만 남김
        nc::NdArray<nc::uint32> item = nc::flatnonzero(iou <= iou_thres);
        if (item.size() != 0)
            order = order(0, item + 1U);
        else
            break;
    }
    pred = pred(nc::asarray(output_idx), pred.cSlice());
    return 0;
}

void clip_coords(nc::NdArray<float> &pred, nc::Shape img_shape)
{
    // MODIFIED for face detection
    // Clip bounding xyxy bounding pred to image shape (height, width)
    nc::NdArray<float> x1 = pred(pred.rSlice(), 0).clip(0.0f, static_cast<float>(img_shape.cols));
    nc::NdArray<float> y1 = pred(pred.rSlice(), 1).clip(0.0f, static_cast<float>(img_shape.rows));
    nc::NdArray<float> x2 = pred(pred.rSlice(), 2).clip(0.0f, static_cast<float>(img_shape.cols));
    nc::NdArray<float> y2 = pred(pred.rSlice(), 3).clip(0.0f, static_cast<float>(img_shape.rows));
    nc::NdArray<float> kps_x1 = pred(pred.rSlice(), 5).clip(0.0f, static_cast<float>(img_shape.cols));
    nc::NdArray<float> kps_y1 = pred(pred.rSlice(), 6).clip(0.0f, static_cast<float>(img_shape.rows));
    nc::NdArray<float> kps_x2 = pred(pred.rSlice(), 7).clip(0.0f, static_cast<float>(img_shape.cols));
    nc::NdArray<float> kps_y2 = pred(pred.rSlice(), 8).clip(0.0f, static_cast<float>(img_shape.rows));
    nc::NdArray<float> kps_x3 = pred(pred.rSlice(), 9).clip(0.0f, static_cast<float>(img_shape.cols));
    nc::NdArray<float> kps_y3 = pred(pred.rSlice(), 10).clip(0.0f, static_cast<float>(img_shape.rows));
    nc::NdArray<float> kps_x4 = pred(pred.rSlice(), 11).clip(0.0f, static_cast<float>(img_shape.cols));
    nc::NdArray<float> kps_y4 = pred(pred.rSlice(), 12).clip(0.0f, static_cast<float>(img_shape.rows));
    nc::NdArray<float> kps_x5 = pred(pred.rSlice(), 13).clip(0.0f, static_cast<float>(img_shape.cols));
    nc::NdArray<float> kps_y5 = pred(pred.rSlice(), 14).clip(0.0f, static_cast<float>(img_shape.rows));
    pred = nc::hstack({
        x1, y1, x2, y2, pred(pred.rSlice(), 4),
        kps_x1, kps_y1, kps_x2, kps_y2, kps_x3, kps_y3, kps_x4, kps_y4, kps_x5, kps_y5
    });
}

void parse_prediction(nc::NdArray<float> &pred, nc::NdArray<int> &bbox, nc::NdArray<float> &conf, nc::NdArray<int> &kps)
{
    bbox = pred(pred.rSlice(), nc::Slice(4)).round().astype<int>();
    conf = pred(pred.rSlice(), 4);
    kps = pred(pred.rSlice(), pred.cSlice(5)).round().astype<int>();
}