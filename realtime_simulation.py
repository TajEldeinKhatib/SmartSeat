import streamlit as st
import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO('C:/Users/Tajkh/PycharmProjects/pythonProject/best (1).pt')
model.classes = [0, 1, 3, 4]  # Only person, objects, availableSeat, and unavailableSeat

# Function to determine seat availability
def determine_seat_availability(detections, frame_width):
    left_person = any(det['class'] == 'person' and det['x1'] < frame_width / 2 for det in detections)
    right_person = any(det['class'] == 'person' and det['x1'] >= frame_width / 2 for det in detections)
    left_object = any(det['class'] == 'objects' and det['x1'] < frame_width / 2 for det in detections)
    right_object = any(det['class'] == 'objects' and det['x1'] >= frame_width / 2 for det in detections)

    left_available = 1 if not left_person and not left_object else 0
    right_available = 1 if not right_person and not right_object else 0

    return left_person, right_person, left_object, right_object, left_available, right_available

# Initialize Streamlit app
st.title('SmartSeat Real-Time Seat Availability')
st.write('This app shows the availability of seats in real-time.')

# Start video capture
cap = cv2.VideoCapture(0)  # Use the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize session state for seat status and timers
if 'left_status' not in st.session_state:
    st.session_state.left_status = []
    st.session_state.left_timer = 0
    st.session_state.left_start_time = None
    st.session_state.left_final_status = 'available'
    st.session_state.left_flicker = False
    st.session_state.left_notes = ''

if 'right_status' not in st.session_state:
    st.session_state.right_status = []
    st.session_state.right_timer = 0
    st.session_state.right_start_time = None
    st.session_state.right_final_status = 'available'
    st.session_state.right_flicker = False
    st.session_state.right_notes = ''

timer_threshold = 0.166 * 60  # 18 seconds threshold in seconds for quicker testing

def update_seat_status(person_detected, object_detected, final_status, start_time, timer, flicker, notes):
    if person_detected:  # Seat is unavailable if a person is detected
        final_status = 'unavailable'
        start_time = None
        timer = 0
        flicker = False
        notes = ''
    elif not person_detected and object_detected:  # Enter pending state
        if final_status != 'pending':
            start_time = time.time()
        final_status = 'pending'
        timer = time.time() - start_time
        if timer > timer_threshold:
            flicker = True
        else:
            flicker = False
        notes = f'**<span style="font-size: 30px;">Time Elapsed: {timer:.2f} seconds</span>**\n\n**<span style="font-size: 30px;">The belongings will be removed from the seat area soon by the library staff</span>**'
    else:  # Seat is available if no person and no objects are detected
        final_status = 'available'
        start_time = None
        timer = 0
        flicker = False
        notes = ''

    return final_status, start_time, timer, flicker, notes

# Function to process and display video frames
def process_video():
    stframe = st.empty()  # Initialize stframe here

    # Create placeholders for seat status
    left_seat_placeholder = st.empty()
    right_seat_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        # Flip the frame horizontally to correct the mirrored effect
        frame = cv2.flip(frame, 1)

        results = model(frame, verbose=False)  # Disable verbose logging for performance

        frame_detections = []
        for result in results:
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

        left_person, right_person, left_object, right_object, left_available, right_available = determine_seat_availability(frame_detections, frame.shape[1])

        st.session_state.left_final_status, st.session_state.left_start_time, st.session_state.left_timer, st.session_state.left_flicker, st.session_state.left_notes = update_seat_status(
            left_person, left_object, st.session_state.left_final_status, st.session_state.left_start_time, st.session_state.left_timer, st.session_state.left_flicker, st.session_state.left_notes)
        st.session_state.right_final_status, st.session_state.right_start_time, st.session_state.right_timer, st.session_state.right_flicker, st.session_state.right_notes = update_seat_status(
            right_person, right_object, st.session_state.right_final_status, st.session_state.right_start_time, st.session_state.right_timer, st.session_state.right_flicker, st.session_state.right_notes)

        # Update seat status in the placeholders
        with left_seat_placeholder.container():
            st.header('Left Seat')
            if st.session_state.left_final_status == 'available':
                st.markdown(f'<div style="background-color:green;padding:10px; font-size: 30px; font-weight: bold;">Status: Available</div>', unsafe_allow_html=True)
                st.session_state.left_notes = ''  # Clear notes when status is available
            elif st.session_state.left_final_status == 'unavailable':
                st.markdown(f'<div style="background-color:red;padding:10px; font-size: 30px; font-weight: bold;">Status: Not Available</div>', unsafe_allow_html=True)
                st.session_state.left_notes = ''  # Clear notes when status is unavailable
            elif st.session_state.left_final_status == 'pending':
                if st.session_state.left_flicker:
                    if int(time.time() * 2) % 2 == 0:
                        st.markdown(f'<div style="background-color:orange;padding:10px; font-size: 30px; font-weight: bold;">Status: Pending</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="background-color:orange;padding:10px; opacity:0.5; font-size: 30px; font-weight: bold;">Status: Pending</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background-color:orange;padding:10px; font-size: 30px; font-weight: bold;">Status: Pending</div>', unsafe_allow_html=True)
                st.write(st.session_state.left_notes, unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="padding:10px; font-size: 30px; font-weight: bold;">&nbsp;</div>', unsafe_allow_html=True)

        with right_seat_placeholder.container():
            st.header('Right Seat')
            if st.session_state.right_final_status == 'available':
                st.markdown(f'<div style="background-color:green;padding:10px; font-size: 30px; font-weight: bold;">Status: Available</div>', unsafe_allow_html=True)
                st.session_state.right_notes = ''  # Clear notes when status is available
            elif st.session_state.right_final_status == 'unavailable':
                st.markdown(f'<div style="background-color:red;padding:10px; font-size: 30px; font-weight: bold;">Status: Not Available</div>', unsafe_allow_html=True)
                st.session_state.right_notes = ''  # Clear notes when status is unavailable
            elif st.session_state.right_final_status == 'pending':
                if st.session_state.right_flicker:
                    if int(time.time() * 2) % 2 == 0:
                        st.markdown(f'<div style="background-color:orange;padding:10px; font-size: 30px; font-weight: bold;">Status: Pending</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="background-color:orange;padding:10px; opacity:0.5; font-size: 30px; font-weight: bold;">Status: Pending</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background-color:orange;padding:10px; font-size: 30px; font-weight: bold;">Status: Pending</div>', unsafe_allow_html=True)
                st.write(st.session_state.right_notes, unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="padding:10px; font-size: 30px; font-weight: bold;">&nbsp;</div>', unsafe_allow_html=True)

        # Display the frame with detected objects
        stframe.image(frame, channels="BGR")

# Button to start and stop video processing
if st.button("Start"):
    process_video()

cap.release()
