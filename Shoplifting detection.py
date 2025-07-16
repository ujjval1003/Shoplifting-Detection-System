import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed, Input

# Define directory paths
base_dir = 'data'
shoplifting_dir = os.path.join(base_dir, 'shoplifting')
normal_dir = os.path.join(base_dir, 'normal')

def extract_frames(video_path, frame_rate=1):
    video = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frame = cv2.resize(frame, (90, 90))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            frames.append(frame)
        count += 1
    video.release()
    return frames

def background_removal(frames):
    bg_removed_frames = []
    for i in range(1, len(frames)):
        diff_frame = cv2.absdiff(frames[i], frames[i-1])
        _, thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
        bg_removed_frames.append(thresh_frame)
    return bg_removed_frames

def shadow_removal(frames):
    shadow_removed_frames = []
    for frame in frames:
        frame[frame < 10] = 0  # Removing shadows by thresholding small pixel values
        shadow_removed_frames.append(frame)
    return shadow_removed_frames

def remove_unimportant_details(frames):
    processed_frames = []
    for frame in frames:
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        processed_frames.append(blurred_frame)
    return processed_frames

def preprocess_video(video_path):
    frames = extract_frames(video_path)
    frames = background_removal(frames)
    frames = shadow_removal(frames)
    frames = remove_unimportant_details(frames)
    return frames

def load_data(video_dir, label):
    data = []
    labels = []
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        frames = preprocess_video(video_path)
        # Divide the frames into sequences of 160 frames
        for i in range(0, len(frames) - 160, 160):
            data.append(frames[i:i+160])
            labels.append(label)
    return np.array(data), np.array(labels)

# Load and preprocess shoplifting videos
shoplifting_data, shoplifting_labels = load_data(shoplifting_dir, 1)

# Load and preprocess normal videos
normal_data, normal_labels = load_data(normal_dir, 0)

# Combine and shuffle the data
X_data = np.concatenate((shoplifting_data, normal_data), axis=0)
y_data = np.concatenate((shoplifting_labels, normal_labels), axis=0)

# Shuffle the data
indices = np.arange(X_data.shape[0])
np.random.shuffle(indices)
X_data = X_data[indices]
y_data = y_data[indices]

# Reshape data to match input shape
X_data = np.expand_dims(X_data, axis=-1)

# Convert labels to categorical
y_data = to_categorical(y_data, num_classes=2)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

def create_cnn(input_shape):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((4, 4)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((4, 4)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    return model

def create_model(input_shape):
    input_frames = Input(shape=input_shape)
    cnn_model = create_cnn((90, 90, 1))
    time_distributed = TimeDistributed(cnn_model)(input_frames)
    lstm_out = LSTM(64, return_sequences=True)(time_distributed)
    lstm_out = LSTM(128)(lstm_out)
    dense_out = Dense(128, activation='relu')(lstm_out)
    dense_out = Dropout(0.25)(dense_out)
    output = Dense(2, activation='softmax')(dense_out)

    model = Model(inputs=input_frames, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define input shape: (sequence_length, height, width, channels)
input_shape = (160, 90, 90, 1)
model = create_model(input_shape)
model.summary()

# Train the Model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {val_accuracy}')

# Save model
model.save('model/lrcn_160S_90_90Q.h5')