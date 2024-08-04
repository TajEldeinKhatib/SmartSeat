import streamlit as st
import cv2
import tempfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('C:/Users/Tajkh/PycharmProjects/pythonProject/best (1).pt')
# names: ["availableSeat","objects","pc","person","unavailableSeat"]
model.classes = [0, 1, 3, 4]  # Only person, objects, availableSeat, and unavailableSeat

st.title('YOLOv8 Video Object Detection')
st.write('Upload a video to detect objects using YOLOv8.')

uploaded_file = st.file_uploader('Choose a video...', type=['mp4', 'avi', 'mov'])
counter = 0
detections = []
frame_width = None  # Variable to store frame width
frame_height = None  # Variable to store frame height

# Timer and status variables
left_timer = 0
right_timer = 0
left_status = 'available'
right_status = 'available'
timer_threshold = 0.2 * 60 * 30  # 15 minutes threshold in frames assuming 30 FPS

def determine_seat_availability(detections, frame_width):
    left_person = any(det['class'] == 'person' and det['x1'] < frame_width / 2 for det in detections)
    right_person = any(det['class'] == 'person' and det['x1'] >= frame_width / 2 for det in detections)
    left_object = any(det['class'] == 'objects' and det['x1'] < frame_width / 2 for det in detections)
    right_object = any(det['class'] == 'objects' and det['x1'] >= frame_width / 2 for det in detections)

    left_available = 1 if not (left_person or left_object) else 0
    right_available = 1 if not (right_person or right_object) else 0

    return left_person, right_person, left_object, right_object, left_available, right_available


# Ground truth data for TEST4 (separate ground truths for left and right seats)
gt_test4_left = [
    (0, 160, 1),  # Left seat is available (1) from frame 0 to 160
    (161, 300, 0)  # Left seat is not available (0) from frame 161 to 300
]

gt_test4_right = [
    (0, 160, 0),  # Right seat is not available (0) from frame 0 to 160
    (161, 300, 1)  # Right seat is available (1) from frame 161 to 300
]

# Ground truth data for TEST2 (separate ground truths for left and right seats)
gt_test2_left = [
    (0, 729, 0),  # Left seat is not available (0) from frame 0 to 729
    (730, 859, 1),  # Left seat is available (1) from frame 730 to 859
    (860, 904, 0),  # Left seat is not available (0) from frame 860 to 904
    (905, 944, 1),  # Left seat is available (1) from frame 905 to 944
    (945, 1355, 0)  # Left seat is not available (0) from frame 945 to 1355
]

gt_test2_right = [
    (0, 729, 0),  # Right seat is not available (0) from frame 0 to 729
    (730, 859, 1),  # Right seat is available (1) from frame 730 to 859
    (860, 904, 0),  # Right seat is not available (0) from frame 860 to 904
    (905, 944, 1),  # Right seat is available (1) from frame 905 to 944
    (945, 1355, 0)  # Right seat is not available (0) from frame 945 to 1355
]


def get_ground_truth(video_name, frame_id, seat_position):
    if video_name == 'TEST4':
        if seat_position == 'left':
            for start, end, label in gt_test4_left:
                if start <= frame_id <= end:
                    return label
        elif seat_position == 'right':
            for start, end, label in gt_test4_right:
                if start <= frame_id <= end:
                    return label
    elif video_name == 'TEST2':
        if seat_position == 'left':
            for start, end, label in gt_test2_left:
                if start <= frame_id <= end:
                    return label
        elif seat_position == 'right':
            for start, end, label in gt_test2_right:
                if start <= frame_id <= end:
                    return label
    return 0  # Default to not available if no match found


if uploaded_file is not None:
    video_name = 'TEST4' if 'TEST4' in uploaded_file.name else 'TEST2'

    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Read the video file
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    # Prepare output video file
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    if cap.isOpened():
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    left_seat_data = []
    right_seat_data = []
    gt_labels_left = []
    gt_labels_right = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if counter % 5 == 0:
            # Perform object detection
            results = model(frame)

            frame_detections = []
            for result in results:
                annotated_frame = result.plot()
                for box in result.boxes:
                    cls = model.names[int(box.cls)]
                    conf = box.conf
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    frame_detections.append({
                        "class": cls,
                        "confidence": conf.item(),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

            detections.append(frame_detections)
            left_person, right_person, left_object, right_object, left_available, right_available = determine_seat_availability(
                frame_detections, frame_width)

            # Update status and timer for left seat
            if left_person:
                left_status = 'unavailable'
                left_timer = 0
            elif not left_person and left_object:
                left_timer += 1
                if left_timer > timer_threshold:
                    left_status = 'available'
                    left_timer = 0
                else:
                    left_status = 'bending/waiting'
            else:
                left_status = 'available'
                left_timer = 0

            # Update status and timer for right seat
            if right_person:
                right_status = 'unavailable'
                right_timer = 0
            elif not right_person and right_object:
                right_timer += 1
                if right_timer > timer_threshold:
                    right_status = 'available'
                    right_timer = 0
                else:
                    right_status = 'bending/waiting'
            else:
                right_status = 'available'
                right_timer = 0

            gt_label_left = get_ground_truth(video_name, counter, 'left')
            gt_label_right = get_ground_truth(video_name, counter, 'right')
            gt_labels_left.append(gt_label_left)
            gt_labels_right.append(gt_label_right)

            left_seat_data.append({
                'frame_id': counter,
                'person': int(left_person),
                'objects': int(left_object),
                'seatAvailability': left_available,
                'groundTruth': gt_label_left,
                'status': left_status,
                'timer': left_timer
            })

            right_seat_data.append({
                'frame_id': counter,
                'person': int(right_person),
                'objects': int(right_object),
                'seatAvailability': right_available,
                'groundTruth': gt_label_right,
                'status': right_status,
                'timer': right_timer
            })

            # Add frame ID annotation
            cv2.putText(annotated_frame, f'Frame ID: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(annotated_frame, f'Left Timer: {left_timer} Status: {left_status}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(annotated_frame, f'Right Timer: {right_timer} Status: {right_status}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Write the frame with the added text to the output video
            out.write(annotated_frame)

            # Display the frame with detected objects and frame ID
            stframe.image(annotated_frame, channels="BGR")

        counter += 1

    cap.release()
    out.release()

    # Provide a download link for the annotated video
    with open(output_file.name, 'rb') as f:
        st.download_button('Download Annotated Video', f, file_name='annotated_video.mp4')

    # Create dataframes
    left_seat_df = pd.DataFrame(left_seat_data)
    right_seat_df = pd.DataFrame(right_seat_data)

    st.write('### Left Seat Detection Results')
    st.dataframe(left_seat_df)

    st.write('### Right Seat Detection Results')
    st.dataframe(right_seat_df)

    # Save dataframes to CSV files
    left_seat_df.to_csv('left_seat_data.csv', index=False)
    right_seat_df.to_csv('right_seat_data.csv', index=False)

    # Provide download links for the CSV files
    with open('left_seat_data.csv', 'rb') as f:
        st.download_button('Download Left Seat Data CSV', f, file_name='left_seat_data.csv')

    with open('right_seat_data.csv', 'rb') as f:
        st.download_button('Download Right Seat Data CSV', f, file_name='right_seat_data.csv')


    # Plot the results
    def plot_seat_data(df, title):
        fig, axs = plt.subplots(4, 1, figsize=(10, 15))
        fig.suptitle(title)

        sns.lineplot(x='frame_id', y='person', data=df, ax=axs[0])
        axs[0].set_title('Person Detection')
        sns.lineplot(x='frame_id', y='objects', data=df, ax=axs[1])
        axs[1].set_title('Object Detection')
        sns.lineplot(x='frame_id', y='seatAvailability', data=df, ax=axs[2])
        axs[2].set_title('Seat Availability')
        sns.lineplot(x='frame_id', y='timer', data=df, ax=axs[3])
        axs[3].set_title('Bending/Waiting Timer')

        st.pyplot(fig)


    plot_seat_data(left_seat_df, 'Left Seat Data')
    plot_seat_data(right_seat_df, 'Right Seat Data')

    # Confusion Matrix and Metrics
    pred_labels_left = left_seat_df['seatAvailability']
    pred_labels_right = right_seat_df['seatAvailability']


    def plot_confusion_matrix(gt_labels, pred_labels, title):
        cm = confusion_matrix(gt_labels, pred_labels)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)


    plot_confusion_matrix(gt_labels_left, pred_labels_left, 'Confusion Matrix for Left Seat')
    plot_confusion_matrix(gt_labels_right, pred_labels_right, 'Confusion Matrix for Right Seat')


    def display_metrics(gt_labels, pred_labels, title):
        accuracy = accuracy_score(gt_labels, pred_labels)
        recall = recall_score(gt_labels, pred_labels)
        precision = precision_score(gt_labels, pred_labels)

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Recall", "Precision"],
            "Score": [accuracy, recall, precision]
        })
        st.write(f'### Evaluation Metrics for {title}')
        st.dataframe(metrics_df)


    display_metrics(gt_labels_left, pred_labels_left, 'Left Seat')
    display_metrics(gt_labels_right, pred_labels_right, 'Right Seat')
