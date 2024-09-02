from flask import Flask, request, jsonify, send_file


import onnxruntime as ort
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the ONNX model
model_path = "best.onnx"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
session = ort.InferenceSession(model_path)

# Retrieve input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type
output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape
# Print input details
print(f"Input Name: {input_name}")
print(f"Input Shape: {input_shape}")
print(f"Input Type: {input_type}")

# Print output details
print(f"Output Name: {output_name}")
print(f"Output Shape: {output_shape}")


# Define a function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_shape[2], input_shape[3]))
    input_data = np.expand_dims(image_resized.transpose(2, 0, 1), axis=0).astype(np.float32)
    return image, input_data


# Define a function to post-process the outputs
def postprocess_output(image, outputs):
    # Assuming the outputs contain bounding boxes and class labels
    # Adjust the unpacking logic based on the actual output structure
    for output in outputs[0]:
        if len(output) == 6:
            x1, y1, x2, y2, conf, cls = output
        else:
            raise ValueError(f"Unexpected output format: {output}")
        if conf > 0.5: # Confidence threshold
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"Class: {int(cls)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
    (0, 255, 0), 2)
    return image


# Define the route to upload an image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Preprocess the image
    original_image, input_data = preprocess_image(file_path)

    # Run inference
    outputs = session.run(None, {input_name: input_data})

    # Post-process the outputs
    processed_image = postprocess_output(original_image, outputs)
    print(f"processed_image: {processed_image}")
    # Save the processed image
    output_path = os.path.join("outputs", file.filename)
    cv2.imwrite(output_path, processed_image)

    return send_file(output_path, mimetype='image/jpeg')


if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    app.run(debug=True)
# curl -X POST -F "file=@cars2.jpg" http://localhost:5000/upload