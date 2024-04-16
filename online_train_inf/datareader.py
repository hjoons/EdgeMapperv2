import os
import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.quantization.quantize import quantize_static, QuantType
from dataloader import OTIDataset

class BufferDataReader(CalibrationDataReader):
    def __init__(self, ds, num) -> None:
        super().__init__()
        self.ds = ds
        self.num = num
        self.cnt = 0

    def get_next(self) -> dict:
        if (self.cnt >= self.num):
            return None
        img = self.ds[self.cnt]['image']
        img_np = np.array(img)
        img_np = np.expand_dims(img_np, axis=0)
        self.cnt += 1
        return {'input.1': img_np}
    
def quantize(infer_path, output_path, tds):
    quant_pre_process(infer_path, infer_path)
    tdr = BufferDataReader(tds, len(tds))
    quantize_static(infer_path, output_path, tdr, weight_type=QuantType.QInt8)