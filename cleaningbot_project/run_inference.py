from tflite_inference.model1_inference import run_inference as run_model1_inference
from tflite_inference.model2_inference import run_inference as run_model2_inference
from camera.camera_feed import capture_frame

def main():
    # Capture frame from the camera
    frame = capture_frame()
    
    if frame is not None:
        # Run inference for model 1
        result1 = run_model1_inference(frame)
        print("Model 1 Inference Result:", result1)

        # Run inference for model 2
        result2 = run_model2_inference(frame)
        print("Model 2 Inference Result:", result2)

if __name__ == "__main__":
    main()
