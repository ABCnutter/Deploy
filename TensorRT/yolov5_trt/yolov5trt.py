#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import os
from datetime import datetime as dt

import calibrator
import cv2
import numpy as np
import tensorrt as trt

from cuda import cudart  # using CUDA Runtime API

# yapf:disable

nHeight = 640
nWidth = 640
onnxFile = "./yolov5s.onnx"
trtFile = "./model.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "\\misc\\"
inferenceImage = dataPath + "corrida-crianças-1280x720.jpg"
dynamic = False
# for FP16 mode
bUseFP16Mode = False
# for INT8 model
bUseINT8Mode = False
nCalibration = 1
cacheFile = "./int8.cache"
calibrationDataPath = dataPath + "test/"
CONFIDENCE_THRESHOLD = 0.4
SCORE_THRESHOLD = 0.2
SCORE_THRESHOLD = 0.2;
NMS_THRESHOLD = 0.4;


# os.system("del ./*.onnx ./*.plan ./*.cache")
np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

# class Rect:
#     def __init__(self, left, top, width, height):
#         self.left = left
#         self.top = top
#         self.width = width
#         self.height = height

class Detect:
    def __init__(self, class_id, confidence, box):
        self.class_id = class_id
        self.confidence = confidence
        self.box = box

def read_txt(txtfile):
    classlist = []
    with open(txtfile, "r") as f:
        for line in f:
            classlist.append(line.strip())
    return classlist

def format_yolov5(source):
    assert source.shape[-1] == 3
    col = source.shape[1];
    row = source.shape[0];
    maxone = max(col, row);
    result = np.zeros([maxone, maxone, 3], dtype=np.float32);
    result[:row, :col, :] = source
    return result

imageorg = cv2.imread(inferenceImage, -1)
print(imageorg.shape)
image = imageorg.astype(np.float32) / 255.0
print(image.shape)
input_image = format_yolov5(image)
x_factor = input_image.shape[1] / nWidth
y_factor = input_image.shape[0] / nHeight

def run(input_img): 
    logger = trt.Logger(trt.Logger.VERBOSE)                                     # Logger, avialable level: VERBOSE, INFO, WARNING, ERRROR, INTERNAL_ERROR
    if os.path.isfile(trtFile):                                                 # read .plan file if exists
        with open(trtFile, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
    else:                                                                       # no .plan file, build engine from scratch
        builder = trt.Builder(logger)                                           # meta data of the network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30                                     # set workspace for TensorRT

        if bUseFP16Mode:
            config.set_flag(trt.BuilderFlag.FP16)
        if bUseINT8Mode:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)

        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnxFile):
            print("Failed finding ONNX file!")
            exit()
        print("Succeeded finding ONNX file!")
        with open(onnxFile, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
            print("Succeeded parsing .onnx file!")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        if dynamic:
            for inputTensor in inputs:
                profile.set_shape(inputTensor.name, [1, 3, nHeight, nWidth], [4, 3, nHeight, nWidth], [8, 3, nHeight, nWidth])
            config.add_optimization_profile(profile)
        # inputTensor = network.get_input(0)
        # profile.set_shape(inputTensor.name, [1, 1, nHeight, nWidth], [4, 1, nHeight, nWidth], [8, 1, nHeight, nWidth])
        # config.add_optimization_profile(profile)
        # last_layer = network.get_layer(network.num_layers - 1)
        network.mark_output(network.get_output(0))
        # network.unmark_output(outputs) 

        engineString = builder.build_serialized_network(network, config)  # create serialized network from the networrk

        if engineString == None:
            print("Failed building serialized engine!")
            return
        print("Succeeded building serialized engine!")
        with open(trtFile, "wb") as f:                                          # save the serialized network as binaray file
            f.write(engineString)
            print("Succeeded saving .plan file!")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)          # create TensorRT engine using Runtime
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")

    # engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    # nIO = engine.num_io_tensors
    # lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    # nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    # context = engine.create_execution_context()
    # context.set_input_shape(lTensorName[0], [1, 3, nHeight, nWidth])
    # for i in range(nIO):
    #     print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    # bufferH = []
    # data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, 3, nHeight, nWidth)
    # bufferH.append(np.ascontiguousarray(data))
    # for i in range(nInput, nIO):
    #     bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    # bufferD = []
    # for i in range(nIO):
    #     bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    # for i in range(nInput):
    #     cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # for i in range(nIO):
    #     context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    # context.execute_async_v3(0)

    # for i in range(nInput, nIO):
    #     cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # for i in range(nIO):
    #     print(lTensorName[i])
    #     print(bufferH[i])

    # for b in bufferD:
    #     cudart.cudaFree(b)

    # print("Succeeded running model in TensorRT!")

    context = engine.create_execution_context()                                 # create CUDA context (similar to a process on GPU)
    if dynamic:
        context.set_binding_shape(0, [1, 3, nHeight, nWidth])      # bind actual shape of the input tensor in Dynamic Shape mode

    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])  # get information of the TensorRT engine
    nOutput = engine.num_bindings - nInput

    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
   
    bufferH = []
    data = np.expand_dims(input_img.transpose(2, 0 ,1), axis=0)
    bufferH.append(np.ascontiguousarray(data))

    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))

    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):                                                     # copy the data from host to device
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)                                                 # do inference computation

    for i in range(nInput, nInput + nOutput):                                   # copy the result from device to host
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nInput + nOutput):
        print("---------------------------------------------")
        print(engine.get_binding_name(i))
        print("---------------------------------------------")
        print(bufferH[i].dtype)
        
    for b in bufferD:                                                           # free the buffer on device
        cudart.cudaFree(b)

    print("Succeeded running model in TensorRT!")

    return bufferH[-1]

from typing import Optional, List

def detect(outs: np.array) -> Detect:
    """YOLOV5后处理

    Args:
        outs (np.array):  Yolov5s模型输出，大小为(1, 25200, 85)

    Returns:
        Detect: YOLOV5经过NMS处理后的输出结果
    """
    
    outs = outs.squeeze(0)
    class_ids = []
    confidences = []
    boxes = []
    output = []
    for i in range(outs.shape[0]):
        confidence = outs[i][4]
        if confidence >= CONFIDENCE_THRESHOLD:
            class_scores = outs[i][5: 85]
            
            _, maxVal, _, maxLoc = cv2.minMaxLoc(np.array(class_scores))
            if maxVal > SCORE_THRESHOLD:
                confidences.append(confidence)
                class_ids.append(maxLoc)

                x = outs[i][0]
                y = outs[i][1]
                w = outs[i][2]
                h = outs[i][3]
                left = (x - 0.5 * w) * x_factor
                top = (y - 0.5 * h) * y_factor
                width = (w * x_factor)
                height = (h * y_factor)
                boxes.append([left, top, width, height])

    nms_result = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

    for i in range(len(nms_result)):
        idx = nms_result[i]
        result = Detect(class_ids[idx], confidences[idx], boxes[idx])
        output.append(result)

    return output

if __name__ == "__main__":
    # os.system("rm -rf ./*.plan")
    # run()                                                                      # create TensorRT engine and do inference
    # run()                                                         # load TensorRT engine from file and do inference

    outs = run(input_img=input_image)
    
    output = detect(outs=outs)

    boxes = output[0].box
    print("-------------------------------------------------------------------",output[0].box)
    cv2.rectangle(imageorg, (int(boxes[0]),int(boxes[1])), (int(boxes[0] + boxes[2]), int(boxes[1] + boxes[3])), (255, 0, 255), 1)
    cv2.imshow("frame", imageorg)
    
    cv2.waitKey()