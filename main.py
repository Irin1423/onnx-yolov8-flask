import onnxruntime as ort
import numpy as np
import cv2
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def predict(name):
    # Use a breakpoint in the code line below to debug your script.
    session = ort.InferenceSession("best.onnx",
                                   providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type

    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    print(f"Input type: {input_type}")

    # Load and preprocess the input image
    image = cv2.imread("input.jpg")
    # print(image.shape)
    image_resized = cv2.resize(image, (input_shape[2], input_shape[3]))
    input_data = np.expand_dims(image_resized.transpose(2, 0, 1), axis=0).astype(np.float32)
    # print(input_data.shape)
    # Run inference
    outputs = session.run(None, {input_name: input_data})

    # Process the outputs (example: print the raw outputs)
    print(outputs)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predict('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
