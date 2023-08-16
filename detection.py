import streamlit as st
import cv2
import numpy as np
from PIL import ImageColor


# Function to perform face detection
def detect_faces(image, scaleFactor, minNeighbors):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(30, 30))
    return faces

def main():
    st.title("Real-time Face Detection App")

    st.write("Welcome to the Real-time Face Detection App!")
    st.write("Instructions:")
    st.write("1. Adjust the sliders to customize face detection parameters.")
    st.write("2. Use the color picker to choose the rectangle color.")
    st.write("3. Click the 'Exit' button to close the app.")
    st.write("4. Detected faces will be highlighted with rectangles on the camera feed.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open camera.")
        return

    exit_button = st.button("Exit")

    # UI elements for customization
    scaleFactor = st.slider("Scale Factor", 1.1, 2.0, 1.2, 0.1)
    minNeighbors = st.slider("Min Neighbors", 1, 10, 5)
    rect_color = st.color_picker("Rectangle Color", "#00FF00")
    st.write(rect_color)

    save_button = st.button("Save Image")

    frame_placeholder = st.empty()  # Create a placeholder for the camera feed

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Could not read a frame from the camera.")
            break

        faces = detect_faces(frame, scaleFactor, minNeighbors)

        for (x, y, w, h) in faces:
            color = ImageColor.getcolor(rect_color, "RGB")
            #st.write(color)

            def hex_to_rgb_normalized(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

            #st.write(hex_to_rgb_normalized(rect_color))

            def compl(rgb_color):
                r, g, b = rgb_color
                return (255 - r, 255 - g, 255 - b)


            cv2.rectangle(frame, (x, y), (x + w, y + h), compl(color), 2)


        frame_placeholder.image(frame, channels="BGR", caption="Camera Feed with Detected Faces", use_column_width=True)

        if save_button:
            cv2.imwrite("detected_faces.jpg", frame)
            st.success("Image saved as 'detected_faces.jpg'")

        if exit_button:
            break

    cap.release()

if __name__ == "__main__":
    main()
