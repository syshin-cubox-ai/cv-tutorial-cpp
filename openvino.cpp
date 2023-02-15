#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "NumCpp.hpp"
#include "scrfd.cpp"

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
        return 1;
    }
    nc::Shape original_img_shape = nc::Shape(img.rows, img.cols);
    img.convertTo(img, CV_32FC3);

    // Inference
    ov::Output input_port = compiled_model.input();
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), img.data);
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
    }
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

    img.convertTo(img, CV_8UC3);
    cv::namedWindow("image");
    cv::imshow("image", img);
    cv::waitKey(0);
    return 0;
}