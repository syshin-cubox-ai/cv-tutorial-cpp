#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "NumCpp.hpp"

using namespace std;

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