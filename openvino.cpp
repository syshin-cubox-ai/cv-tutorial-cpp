#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "NumCpp.hpp"
#include "scrfd.hpp"

using namespace std;

int draw_prediction(cv::Mat &img, nc::NdArray<int> &bbox, nc::NdArray<float> &conf, nc::NdArray<int> &kps, int thickness=2, bool hide_conf=false)
{
    // Draw prediction on the image. If the keypoints is None, only draw the bbox.
    if (!((bbox.numRows() == conf.numRows()) && (conf.numRows() == kps.numRows())))
    {
        cerr << "bbox, conf, and kps must be equal length." << endl;
        return -1;
    }

    cv::Scalar bbox_color = cv::Scalar(0, 255, 0);
    cv::Scalar conf_color = cv::Scalar(0, 255, 0);
    cv::Scalar kps_colors[5] = {cv::Scalar(0, 165, 255), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255)};
    for (int i = 0; i < bbox.numRows(); i++)
    {
        // Draw bbox
        cv::rectangle(img, cv::Point(bbox(i, 0), bbox(i, 1)), cv::Point(bbox(i, 2), bbox(i, 3)), bbox_color, thickness, cv::LINE_AA);

        // Text confidence
        cv::putText(img, to_string(conf(i, 0)).substr(0, 3), cv::Point(bbox(i, 0), bbox(i, 1) - 2), cv::FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, thickness, cv::LINE_AA);

        // Draw keypoints
        for (int j = 0, k = 0; j < 10; j += 2, k++)
        {
            cv::circle(img, cv::Point(kps(i, j), kps(i, j + 1)), thickness, kps_colors[k], cv::FILLED);
        }
    }
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

    draw_prediction(img, bbox, conf, kps);

    img.convertTo(img, CV_8UC3);
    cv::namedWindow("image");
    cv::imshow("image", img);
    cv::waitKey(0);
    return 0;
}