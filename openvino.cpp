#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

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
    ov::Tensor output = infer_request.get_output_tensor();
    float *output_data = output.data<float>();
    // output_data[] - accessing output tensor data
    ov::Shape output_shape = output.get_shape();
    size_t output_size = output.get_size();
    vector<float> result(output_data, output_data + output_size);
    return 0;
}