import time
import numpy as np
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate

# Replace 'model.tflite' with the path to your model file
model_path = '/home/mendel/safety-vision/model/model_edgetpu.tflite'

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up a sample input
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)

# Warm-up runs (optional)
for _ in range(5):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

# Run the benchmark
num_runs = 50
start_time = time.time()

for _ in range(num_runs):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

elapsed_time = time.time() - start_time

# Calculate the average inference latency
average_latency = elapsed_time / num_runs * 1000
print(f'Average inference latency for {num_runs} runs: {average_latency:.2f} ms')