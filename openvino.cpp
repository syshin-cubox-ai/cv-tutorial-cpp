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
        nc::NdArray<float> inter_x1 = nc::maximum(nc::repeat(x1(0, highest_conf_item), static_cast<nc::uint32>(1), rest_item.shape().cols), x1(0, rest_item));
        nc::NdArray<float> inter_y1 = nc::maximum(nc::repeat(y1(0, highest_conf_item), static_cast<nc::uint32>(1), rest_item.shape().cols), y1(0, rest_item));
        nc::NdArray<float> inter_x2 = nc::minimum(nc::repeat(x2(0, highest_conf_item), static_cast<nc::uint32>(1), rest_item.shape().cols), x2(0, rest_item));
        nc::NdArray<float> inter_y2 = nc::minimum(nc::repeat(y2(0, highest_conf_item), static_cast<nc::uint32>(1), rest_item.shape().cols), y2(0, rest_item));
        nc::NdArray<float> w = nc::maximum(nc::zeros<float>(1, rest_item.shape().cols), inter_x2 - inter_x1);
        nc::NdArray<float> h = nc::maximum(nc::zeros<float>(1, rest_item.shape().cols), inter_y2 - inter_y1);
        nc::NdArray<float> intersection = w * h;
        nc::NdArray<float> o_union = nc::repeat(area(0, highest_conf_item), static_cast<nc::uint32>(1), rest_item.shape().cols) + area(0, rest_item) - intersection;
        nc::NdArray<float> iou = intersection / o_union;

        // 임계값 이하로 iou를 가진 항목만 남김
        nc::NdArray<nc::uint32> item = nc::flatnonzero(iou <= iou_thres);
        if (item.size() != 0)
            order = order(0, item + static_cast<nc::uint32>(1));
        else
            break;
    }
    pred = pred(nc::asarray(output_idx), pred.cSlice());
    return 0;
}

int main()
{
    // Load model
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model("../openvino_files/scrfd_2.5g_bnkps.xml", "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Load image
    cv::Mat img;
    img = cv::imread("../img/2.jpg");
    if (img.empty())
    {
        cerr << "Image load failed." << endl;
        return 1;
    }
    img.convertTo(img, CV_32FC3);

    // Get input port for model with one input
    ov::Output input_port = compiled_model.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), img.data);
    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);
    // Start inference
    infer_request.infer();
    // Get output tensor for model with one output
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    size_t output_size = output_tensor.get_size();
    float *output_data = output_tensor.data<float>();
    // output_data[] - accessing output tensor data
    nc::NdArray<float> pred = nc::asarray(output_data, output_shape[0], output_shape[1]);

    int retval = non_max_suppression(pred, 0.3f, 0.5f);
    if (retval != 0)
    {
        cout << "No faces detected." << endl;
    }
    else
    {
        cout << "pred:\n" << pred << endl;
    }
    return 0;
}