from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import subprocess
import cv2
import numpy as np
import tensorflow as tf
import math
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_PATH'] = 'model/lrcn_160S_90_90Q.h5'

class ShopliftingPrediction:
    def __init__(self, model_path, frame_width, frame_height, sequence_length):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def generate_message_content(self, probability, label):
        if label == 0:
            if probability <= 75:
                self.message = "There is little chance of theft"
            elif probability <= 85:
                self.message = "High probability of theft"
            else:
                self.message = "Very high probability of theft"
        elif label == 1:
            if probability <= 75:
                self.message = "The movement is confusing, watch"
            elif probability <= 85:
                self.message = "I think it's normal, but it's better to watch"
            else:
                self.message = "Movement is normal"

    def Pre_Process_Video(self, current_frame, previous_frame):
        diff = cv2.absdiff(current_frame, previous_frame)
        diff = cv2.GaussianBlur(diff, (3, 3), 0)
        resized_frame = cv2.resize(diff, (self.frame_height, self.frame_width))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = gray_frame / 255
        return normalized_frame

    def Read_Video(self, filePath):
        self.video_reader = cv2.VideoCapture(filePath)
        self.original_video_width = int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_video_height = int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video_reader.get(cv2.CAP_PROP_FPS)

    def Single_Frame_Predict(self, frames_queue):
        probabilities = self.model.predict(np.expand_dims(frames_queue, axis=0))[0]
        predicted_label = np.argmax(probabilities)
        probability = math.floor(max(probabilities[0], probabilities[1]) * 100)
        return [probability, predicted_label]

    def Predict_Video(self, video_file_path, output_file_path):
        self.Read_Video(video_file_path)
        video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                       self.fps, (self.original_video_width, self.original_video_height))
        success, frame = self.video_reader.read()
        previous = frame.copy()
        frames_queue = []
        message = 'I will start analysis video now'
        while self.video_reader.isOpened():
            ok, frame = self.video_reader.read()
            if not ok:
                break
            normalized_frame = self.Pre_Process_Video(frame, previous)
            previous = frame.copy()
            frames_queue.append(normalized_frame)
            if len(frames_queue) == self.sequence_length:
                [probability, predicted_label] = self.Single_Frame_Predict( frames_queue)
                self.generate_message_content(probability, predicted_label)
                message = "{}:{}%".format(self.message, probability)
                frames_queue = []
            cv2.rectangle(frame, (0, 0), (640, 40), (255, 255, 255), -1)
            cv2.putText(frame, message, (1, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            video_writer.write(frame)
        self.video_reader.release()
        video_writer.release()

def convert_video(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        output_path
    ]
    subprocess.run(command, check=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            unique_id = uuid.uuid4().hex
            input_filename = f"{unique_id}_input.mp4"
            output_filename = f"{unique_id}_output.mp4"
            converted_output_filename = f"{unique_id}_output_converted.mp4"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            converted_output_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_output_filename)

            file.save(input_path)

            # Instantiate the model and run the prediction
            model_path = app.config['MODEL_PATH']
            predictor = ShopliftingPrediction(model_path, 90, 90, 160)
            predictor.Predict_Video(input_path, output_path)

            # Convert video to a web-friendly format
            convert_video(output_path, converted_output_path)

            return redirect(url_for('result', output_filename=converted_output_filename))
    return render_template('upload.html')

@app.route('/processing')
def processing():
    return render_template('processing.html')

@app.route('/result')
def result():
    output_filename = request.args.get('output_filename')
    return render_template('result.html', output_filename=output_filename)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)