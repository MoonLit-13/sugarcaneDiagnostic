import picamera
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import time

# Class names (must match training order)
class_names = ['BacterialBlights', 'Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# Load TFLite model
interpreter = tflite.Interpreter(model_path='sugarcane_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to capture and classify
def classify_leaf():
    # Capture image
    with picamera.PiCamera() as camera:
        camera.resolution = (1024, 768)
        camera.start_preview()
        time.sleep(2)  # Warm up camera
        camera.capture('temp_image.jpg')
        camera.stop_preview()

    # Load and preprocess image
    img = Image.open('temp_image.jpg')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction
    predicted_index = np.argmax(output)
    predicted_class = class_names[predicted_index]
    confidence = output[0][predicted_index]

    print(f'Predicted Disease: {predicted_class}')
    print(f'Confidence: {confidence:.4f}')

    # Optional: Display probabilities
    print('All Probabilities:', output[0])

if __name__ == '__main__':
    classify_leaf()