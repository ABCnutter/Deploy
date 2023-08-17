#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>


//��class���ص�vector��
std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("config_files/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        //std::cout << "Class:---------------:::::::" << line << std::endl;

        class_list.push_back(line);
    }
    return class_list;
}

// ��������
void load_net(cv::dnn::Net &net, bool is_cuda)
{
    //ʹ��readNet������������YOLOV5S.ONNX�ļ�
    auto result = cv::dnn::readNet("config_files/yolov5s.onnx");

    //�������ѡ���Ƿ�ʹ��CUDA
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

//���������ɫ
const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};


// ������ز���ֵ����ֵ
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

// �����������Ľṹ����
struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

//�������ͼ�����Ԥ��������һ����ʽ�����ͼ��
cv::Mat format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}


//YOLOV5���������Ԥ�����Լ�ǰ����������NMS����
void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    cv::Mat blob;

    auto input_image = format_yolov5(image);
    
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);  // blob-[3,640,640]
    std::vector<cv::Mat> outputs;

    //������㵽ָ���㣨�ڶ�������ָ���Ĳ㣩�������ظò���������
    net.forward(outputs, net.getUnconnectedOutLayersNames()); // getUnconnectedOutLayersNames()���ؾ���δ��������Ĳ������,�������������

    //����x_factor��y_factor�����ں��滹ԭbounding box��λ�úʹ�С
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    /*
    ͨ��outputs[0]���Ի�ø������Ľ�������а����˸ò����е�Ԥ������Ϣ������Ԥ����λ�á���С�����ŶȺ������ʡ�
    ��Щ��Ϣ��������һ��ָ�������ڴ�ĵ�ַ�У�����ͨ��.data�����ʡ�

    outputs[0].data����һ��ָ��float���͵������ڴ��ָ�룬����ָ��ָ�����һ��float���͵����飬���а����˸ò�����Ԥ����λ�á���С�����ŶȺ������ʡ�
    ��ˣ�����ָ�븳ֵ��float* data�󣬾Ϳ���ͨ��data�����ʸ������е�ÿһ��Ԫ�ء�
    ͬʱ�����ڸ������������ڴ棬��˿���ͨ��ָ����������������ʸ������е�ÿһ��Ԫ�أ���ʹ��data[i]�����������е�i��Ԫ��
    */
    float *data = (float *)outputs[0].data;
    //std::cout << "---------------------------------------:" << sizeof(&data) << std::endl;
    /*
    Yolov5sģ�͵������СΪ(1, 25200, 85)�����У�

    ��һά��batch size��Ϊ1��
    �ڶ�άΪÿ������ͼƬ���ɵ�Ԥ���������anchors���� x (S1 x S1 + S2 x S2 + S3 x S3)�������S1, S2, S3�ֱ�Ϊ��������������ͼ�Ĵ�С��ȡֵΪ{80, 40, 20}��anchors����Ϊ3������ܵ�Ԥ�����Ϊ25200��
    ����άΪÿ��Ԥ������Ϣ������4��������Ϣ��1�����Ŷ���Ϣ��80�����÷���Ϣ����85����Ϣ��
    */
    const int dimensions = 85;
    const int rows = 25200;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        /*
        ��C++�У�ָ��ʹ��[]������ʱ�����������������ơ�
        ��ָ��ָ������������ڴ�����ʱ������ʹ��[]�����������ʸ������е����ݡ�
        ���磬�����һ��ָ��float�������ݵ�ָ��p�����ǿ���ͨ��p[i]����������ָ����ڴ��еĵ�i��float���͵����ݡ�
        �����i��һ������������ָ����Ҫ���ʵ��������ڴ��е�ƫ������

        data[4]��data + 5���Էֱ����ָ����ָ����ڴ��еĵ�5��float���͵����ݺʹӵ�6��float���͵����ݿ�ʼ��һ���������ݡ�
        */
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float * classes_scores = data + 5;
            /*
            cv::Mat�Ĺ��캯���������£�
            ��һ���������������������������Ϊ1����ʾֻ��һ�У�
            �ڶ����������������������������ΪclassName.size()����ʾ��className.size()�У�
            ������������������������ͣ���������ΪCV_32FC1����ʾԪ������Ϊ��ͨ��32λ��������
            ���ĸ����������������ָ�룬��������Ϊclasses_scores����ʾ��������ݴ洢��classes_scores��ָ����ڴ��ַ����ָ������ڴ���׵�ַ��
            */
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;

            //��ȡ����������Լ����Ӧ������
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            //ͨ����ֵ����ɸѡ��������Ҫ���������Ŷ��Լ�������б���
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
        //һ���߽�����85��ֵ��������4��������Ϣ��1�����Ŷ���Ϣ��80�����÷���Ϣ
        //data��ָ�ڴ��ַ�������������Ԥ����λ�á���С�����ŶȺ������ʣ���yolov5s�й���25200���߽�򣬼�data��ָ�ڴ��ַ����25200*85��ֵ
        //�ڱ���һ���߽���dataָ����Ҫ����ƶ�85��λ�ã��� +85
        data += 85;

    }

    std::vector<int> nms_result;
    /*
    ��Ŀ���������У�һ��Ŀ����ܻᱻ����߽���⵽����Щ�߽����ܻ��в�ͬ��λ�úʹ�С������ʾͬһ��Ŀ�ꡣ
    �Ǽ���ֵ���ƣ�Non-Maximum Suppression��NMS����һ�ֳ��õķ���������������Щ�ص��ı߽��ֻ�������Ŷ���ߵ��Ǹ��߽�򣬴Ӷ��õ����յ�Ŀ��������

    NMS��ԭ�����£�

    ���ȣ������еı߽���������ŶȽ����������Ŷ���ߵı߽��������ǰ�档

    �����Ŷ���ߵı߽��ʼ�����α�������߽��

    ���ڵ�ǰ�������ı߽���������ǰ���Ѿ������ı߽����ص��̶ȣ�ͨ������IOUֵ������һ����ֵ������0.5������ô�ͽ������Ƶ�����������

    ����������һ���߽���ظ��������̣�ֱ�����еı߽�򶼱�������ϡ�

    ͨ�������Ĵ���NMS�������Ƶ������ص��ı߽��ֻ������õ��Ǹ��߽�򣬴Ӷ��õ����յ�Ŀ�����������ַ�����Ȼ�򵥣�������ʵ���зǳ���Ч���Ѿ����㷺Ӧ���ڸ���Ŀ���������С�
    
    ���ڷǼ���ֵ���ƣ������µ�һƪ��������ϸ���⣬�������ǲ��������������ο���
    */
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    
    //������NMS�����Ľ�����ص�const vector<Detection> output��
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

int main(int argc, char **argv)
{
    // ����class�б�
    std::vector<std::string> class_list = load_class_list();

    //��ȡ��Ƶ�ļ����ж��Ƿ�ɹ���
    cv::Mat frame;
    cv::VideoCapture capture("sample.mp4");
    if (!capture.isOpened())
    {
        std::cerr << "Error opening video file\n";
        return -1;
    }

    //cv::Mat frame = cv::imread("misc/araras.jpg");
    //�ж��Ƿ�ʹ��CUDA,��ѡ��ʹ��CUDAǰ����ȷ������֧��GPU�Լ���װ��CUDA��cudnn��
    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    
    std::cout << "---------------is_cuda is : ("<<is_cuda<<")------------------------" << std::endl;

    //����YOLOV5����
    cv::dnn::Net net;
    load_net(net, is_cuda);

    //�����߾��ȼ�ʱ��
    auto start = std::chrono::high_resolution_clock::now();
    //��¼��Ƶ֡��
    int frame_count = 0;
    //����FPS(ÿ�봫��֡��)
    float fps = -1;
    int total_frames = 0;

    while (true)
    {
        //��ȡ��Ƶ֡, ���ڹ�����Ƶ����Ӧ��Ҳ�ɲο���������opencv��������
        capture.read(frame);
        if (frame.empty())
        {
            std::cout << "End of stream\n";
            break;
        }

        std::vector<Detection> output;
        //YOLOV5Sǰ������
        detect(frame, net, output, class_list);

        frame_count++;
        total_frames++;
        
        //���ı߽������
        int detections = output.size();

        for (int i = 0; i < detections; ++i)
        {

            auto detection = output[i];

            auto box = detection.box;
            auto classId = detection.class_id;

            //ͨ��ȡģ����Ϊ�߽��ѡ����ɫ
            const auto color = colors[classId % colors.size()];

            //���Ʊ߽��
            cv::rectangle(frame, box, color, 3);
            //��������д���ı߿�Χ��һ����ڱ߿������
            cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            //��������ƵĿ����д�����
            cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        //����֡���Լ���ʱ���������FPS
        if (frame_count >= 30)
        {

            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        //���FPS����0��������Ƶ���Ͻ�д����
        if (fps > 0)
        {

            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("output", frame);

        //����û������˼�����ôcapture.release()�������ͷ���Ƶ�����������ѭ�����ر���Ƶ����ͬʱ���������һ����Ϣ��ʾ�û������Ѿ�������
        if (cv::waitKey(1) != -1)
        {
            capture.release();
            std::cout << "finished by user\n";
            break;
        }
    }

    //�����Ƶ������֡��
    std::cout << "Total frames: " << total_frames << "\n";

    return 0;
}