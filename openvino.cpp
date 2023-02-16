#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "NumCpp.hpp"
#include "scrfd.hpp"

using namespace std;

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
        return -1;
    }
    nc::Shape original_img_shape = nc::Shape(img.rows, img.cols);
    float scale;
    cv::Mat transformed_img = img.clone();
    transform_image(transformed_img, scale);

    // Inference
    ov::Output input_port = compiled_model.input();
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), transformed_img.data);
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    size_t output_size = output_tensor.get_size();
    float *output_data = output_tensor.data<float>();
    nc::NdArray<float> pred = nc::asarray(output_data, output_shape[0], output_shape[1]);
    int retval = non_max_suppression(pred, 0.3f, 0.5f);
    if (retval != 0)
    {
        cout << "No faces detected." << endl;
        return 0;
    }
    pred = nc::hstack({
        pred(pred.rSlice(), nc::Slice(4)) / scale,
        pred(pred.rSlice(), 4), pred(pred.rSlice(),
        pred.cSlice(5)) / scale
    });
    clip_coords(pred, original_img_shape);
    nc::NdArray<int> bbox;
    nc::NdArray<float> conf;
    nc::NdArray<int> kps;
    parse_prediction(pred, bbox, conf, kps);

    // Draw prediction
    cv::rectangle(img, cv::Point(bbox(0, 0), bbox(0, 1)), cv::Point(bbox(0, 2), bbox(0, 3)), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::rectangle(img, cv::Point(bbox(1, 0), bbox(1, 1)), cv::Point(bbox(1, 2), bbox(1, 3)), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::putText(img, to_string(conf(0, 0)), cv::Point(bbox(0, 0), bbox(0, 1) - 2),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA
    );
    cv::putText(img, to_string(conf(1, 0)), cv::Point(bbox(1, 0), bbox(1, 1) - 2),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA
    );
    cv::Scalar kps_colors[] = {cv::Scalar(0, 165, 255), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255)};
    cv::circle(img, cv::Point(kps(0, 0), kps(0, 1)), 2, kps_colors[0], cv::FILLED);
    cv::circle(img, cv::Point(kps(0, 2), kps(0, 3)), 2, kps_colors[1], cv::FILLED);
    cv::circle(img, cv::Point(kps(0, 4), kps(0, 5)), 2, kps_colors[2], cv::FILLED);
    cv::circle(img, cv::Point(kps(0, 6), kps(0, 7)), 2, kps_colors[3], cv::FILLED);
    cv::circle(img, cv::Point(kps(0, 8), kps(0, 9)), 2, kps_colors[4], cv::FILLED);
    cv::circle(img, cv::Point(kps(1, 0), kps(1, 1)), 2, kps_colors[0], cv::FILLED);
    cv::circle(img, cv::Point(kps(1, 2), kps(1, 3)), 2, kps_colors[1], cv::FILLED);
    cv::circle(img, cv::Point(kps(1, 4), kps(1, 5)), 2, kps_colors[2], cv::FILLED);
    cv::circle(img, cv::Point(kps(1, 6), kps(1, 7)), 2, kps_colors[3], cv::FILLED);
    cv::circle(img, cv::Point(kps(1, 8), kps(1, 9)), 2, kps_colors[4], cv::FILLED);

    img.convertTo(img, CV_8UC3);
    cv::namedWindow("image");
    cv::imshow("image", img);
    cv::waitKey(0);
    return 0;
}