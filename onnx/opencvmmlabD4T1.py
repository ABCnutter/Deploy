import torch 
import numpy as np
import onnx, onnxruntime


class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, x): 
        return torch.asinh(x) 
 
# from torch.onnx.symbolic_registry import register_op 
 
# def asinh_symbolic(g, input, *, out=None): 
#     return g.op("Asinh", input) 
 
# register_op('asinh', asinh_symbolic, '', 9) 
 
model = Model() 
input = torch.rand(1, 3, 10, 10) 
# torch.onnx.export(model, input, 'asinh.onnx', opset_version=9, input_names=["inputs"], output_names=["outputs"]) 

torch_outs = model(input).detach().numpy()

sess = onnxruntime.InferenceSession("asinh.onnx")
ort_outs = sess.run(["outputs"], {"inputs": input.numpy()})[0]

assert np.allclose(torch_outs, ort_outs) 

from pprint import pprint

pprint(f"torch outs is :{torch_outs}")
pprint(f"ort outs is :{ort_outs}")
