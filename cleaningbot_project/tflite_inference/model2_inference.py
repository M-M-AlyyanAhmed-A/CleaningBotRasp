import tflite_runtime.interpreter as tflite
from camera.utils import preprocess_image


def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

if __name__ == "__main__":
    model_path = "/home/pi/tflite_inference/models/2-best-fp16.tflite"
    interpreter = load_model(model_path)

    # Capture frame and preprocess it
    from camera.camera_feed import capture_frame
    frame = capture_frame()

    if frame is not None:
        input_data = preprocess_image(frame)
        result = run_inference(interpreter, input_data)
        print("Inference Result:", result)
