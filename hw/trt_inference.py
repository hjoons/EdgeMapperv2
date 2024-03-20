import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def trt_to_np(trt_type):
	"""
	Helper function for trt.DataType to np datatype conversion
	
	trt library already has nptype function, but numpy (1.24.4) bool deprecated so
	function no longer works
	
	Parameters:
		trt_type - dtype of trt tensor binding
	
	Returns:
		Equivalent np datatype OR bool
	"""
	if trt_type == trt.DataType.FLOAT:
		return np.float32
	elif trt_type == trt.DataType.HALF:
		return np.float16
	elif trt_type == trt.DataType.INT32:
		return np.int32
	elif trt_type == trt.DataType.INT8:
		return np.int8
	else:
		return bool
	

class trtModel():
	"""
	This class creates an object for TensorRT inference
	"""
	
	def __init__(self, model_pth: str):
		"""
		Initializes trtModel object and creates necessary buffers
		
		Parameters:
			model_pth:str - file path for TensorRT model
		"""
		f = open(model_pth, 'rb')
		runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
		
		engine = runtime.deserialize_cuda_engine(f.read())
		self.context = engine.create_execution_context()
		
		in_type = trt_to_np(engine.get_tensor_dtype(engine[0]))
		out_type = trt_to_np(engine.get_tensor_dtype(engine[1]))
		
		in_shape = self.context.get_tensor_shape(engine[0])
		self.out_shape = self.context.get_tensor_shape(engine[1])
		
		self.h_input = cuda.pagelocked_empty(trt.volume(in_shape), dtype=in_type)
		self.h_output = cuda.pagelocked_empty(trt.volume(self.out_shape), dtype=out_type)
		self.d_input = cuda.mem_alloc(self.h_input.nbytes)
		self.d_output = cuda.mem_alloc(self.h_output.nbytes)
		self.bindings = [int(self.d_input), int(self.d_output)]
		
		self.stream = cuda.Stream()
		f.close()
	
	def inference(self, batch):
		"""
		Performs forward pass of tensorrt model
		
		Parameters:
			batch - numpy array with input, dimensions should match the input tensor shape
			
		Returns:
			numpy array of output prediction with all dims
		"""
		# transfer input data to device
		np.copyto(self.h_input, batch.ravel())
		cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
		# execute model
		self.context.execute_async_v2(self.bindings, self.stream.handle)
		# transfer output back
		cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
		# synchronize threads
		self.stream.synchronize()
		
		return self.h_output
		
	def reload_model(self, model_pth: str):
		"""
		Reload with new TensorRT model
		
		Creates a new trt execution context for a new model
		New model should have same input and output datatypes and shapes, doesn't check
		Will use the same input and output buffers previously created
		
		Parameters:
			model_pth:str - file path for new TensorRT model
		"""
		f = open(model_pth, 'rb')
		runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
		engine = runtime.deserialize_cuda_engine(f.read())
		self.context = engine.create_execution_context()
		f.close()

# example showing how to use trtModel class
if __name__ == "__main__":
	from h5dataset import createH5TestLoader
	from time import perf_counter
	from matplotlib import pyplot as plt
	
	# get an input image
	test_loader = createH5TestLoader("../comm-deploy/pma_escalator.h5")
	for batch_idx, batch in enumerate(test_loader):
		image = batch['image'].numpy()
		break
	
	# initialize model with fp32 model
	model = trtModel('mbnv3.trt')
	
	# benchmark model
	times = []
	for num in range(1000):
		t_start = perf_counter()
		pred = model.inference(image)
		t_stop = perf_counter()
		if num > 4:
			times.append(t_stop - t_start)
		
	print(f'Avg per inference (after warming up): {np.mean(times)}')
	plt.imshow(pred[0][0])
	plt.title('TensorRT fp32 Inference')
	plt.show()
	
	# reload with fp16 model
	# reloading can be used when receiving a new global model via federated learning since it will have the same input and output shape and type
	model.reload_model('mbnv3_fp16.trt')
	
	# benchmark model
	times = []
	for num in range(1000):
		t_start = perf_counter()
		pred = model.inference(image)
		t_stop = perf_counter()
		if num > 4:
			times.append(t_stop - t_start)
		
	print(f'Avg per inference (after warming up): {np.mean(times)}')
	plt.imshow(pred[0][0])
	plt.title('TensorRT fp16 Inference')
	plt.show()
		

