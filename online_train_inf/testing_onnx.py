import  onnxruntime as ort


print(f"onnxruntime device: {ort.get_device()}") # output: GPU

print(f'ort avail providers: {ort.get_available_providers()}') # output: ['CUDAExecutionProvider', 'CPUExecutionProvider']

ort_session = ort.InferenceSession('temp.onnx', providers=["CUDAExecutionProvider"])

print(ort_session.get_providers()) # output: ['CPUExecutionProvider']
