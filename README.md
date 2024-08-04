# SmartSeat Project

## Introduction
SmartSeat is an innovative project designed to enhance the efficiency of university libraries by detecting seat availability in real-time. Our system uses advanced cameras, machine learning, deep learning, and computer vision to analyze the presence of people and their belongings. This ensures accurate, real-time updates on seat occupancy, helping students effortlessly find open study spaces and optimizing library resource management.

## Features
- Real-Time Seat Detection: Detects whether seats are occupied by people or belongings.
- Performance Evaluation: Interactive page to evaluate the model's performance with various metrics.
- Simulation and Demo: Real-time simulation of seat availability, displaying the status using color-coded indicators.

## Project Structure
- `evaluation.py`: Streamlit script for evaluating the model performance.
- `realtime_simulation.py`: Streamlit script for real-time simulation and demo of the SmartSeat system.
- `yolov8n.pt`: File containing pre-computed predictions used for evaluation.

## Installation
To run this project, you'll need to have Python installed on your machine. You can install the necessary packages using the following command:

pip install -r requirements.txt

## Usage

### Running the Evaluation Script
1. Open a terminal and navigate to the project directory:
   cd path/to/your/project
2. Start the evaluation page:
   streamlit run evaluation.py
3. Upload your dataset and view the performance metrics and confusion matrices on the Streamlit interface.

### Running the Real-Time Simulation Script
1. Open a terminal and navigate to the project directory:
   cd path/to/your/project
2. Start the real-time simulation page:
   streamlit run realtime_simulation.py
3. Observe the real-time seat availability status for the right and left seats using the color-coded indicators on the Streamlit interface.


## License
This project is private and not currently available under an open-source license.

## Contact
For any inquiries or support, please contact:
- Taj Eldein Khatib
- Majd Eldein Zreeke

Thank you for your interest in SmartSeat!