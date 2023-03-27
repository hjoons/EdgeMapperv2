from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

class QuantizationDataReader(CalibrationDataReader):
    def __init__(self, batch_size, input_name):
        self.torch_dl = get_loader(zipfile='/home/orin/Documents/FH12_23-24/EdgeMapperv2/hw/nyu.zip', batch_size=1, split='eval')
        
        self.input_name = input_name
        self.datasize = len(self.torch_dl)
        
        self.enum_data = iter(self.torch_dl)
       
    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()
    
    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
            batch = batch['image']
            return {self.input_name: self.to_numpy(batch)}
        else:
            return None
        
        
        def rewind(self):
            self.enum_data = iter(self.torch_dl)

qdr = QuantizationDataReader(batch_size=1, input_name=sess.get_inputs()[0].name)

quantize_static(f'{model_name}.onnx', f'./{model_name}_int8.onnx', weight_type=QuantType.QInt8, calibration_data_reader=qdr)
