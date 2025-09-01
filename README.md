# Stroke Detection Dashboard 🧠

This project is an interactive web dashboard built with Streamlit that uses a deep learning model to detect early signs of facial asymmetry associated with a stroke from an image. The application allows users to analyze images either from a file upload or a live camera feed.

This dashboard was developed as a final project for a B.Sc. in Information Systems Engineering, specializing in Data Science.

![Dashboard Preview](https://i.postimg.cc/1zHhDSRX/dashboard-preview.png)

---
## Features
* **Live Prediction Tool:** Analyze images from a file upload or live camera feed using a stateful, multi-screen interface.
* **Dual Model Analysis:** Choose between a robust Ensemble model (recommended) or the best-performing single model (ResNet50V2).
* **Comprehensive Model Insights:** A dedicated section with three sub-tabs for a complete performance analysis:
    * **Performance Summary:** High-level KPIs, model score comparisons, and a final recommendation.
    * **In-Depth Comparison:** Side-by-side plots (Confusion Matrix, ROC Curve, PR Curve) for the two main models.
    * **Interactive Explorer:** A tool to explore the decision threshold's impact on metrics in real-time.
* **Project Documentation:** Includes detailed sections about the project's goal, methodology, and contact information.

## Tech Stack
* **Language:** Python
* **App Framework:** Streamlit
* **Deep Learning:** TensorFlow / Keras
* **Data & Plotting:** NumPy, Pandas, Matplotlib, Seaborn
* **Image Processing:** OpenCV, Pillow, Albumentations

## File Structure
The project is organized into the following directories:
```
stroke_dashboard/
│
├── .streamlit/
│   └── config.toml       # Streamlit theme configuration
│
├── files/
│   ├── deploy.prototxt   # Face detector model structure
│   └── res10_300x300_ssd.caffemodel  # Face detector model weights
│
├── images/
│   ├── logo.png          # App icon and loading screen image
│   └── *.jpg             # Profile pictures for the contact page
│
├── models/
│   └── *.keras           # Saved deep learning models
│
├── ynpy/
│   └── *.npy             # Saved arrays for the interactive explorer
│
├── config.json           # Main configuration for models and thresholds
├── stroke_detection_dashboard.py  # The main Streamlit application script
└── requirements.txt      # Required Python libraries
```

## Setup & Installation
To run this dashboard locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tomer-pinto/stroke-detection-dashboard.git
    cd stroke_dashboard
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run stroke_detection_dashboard.py
    ```

---
## Contact
* **Tomer Pinto:** [LinkedIn](https://www.linkedin.com/in/tomerpinto/) | [GitHub](https://github.com/tomer-pinto)
* **Tomer Amon:** [LinkedIn](https://www.linkedin.com/in/tomer-amon-9aa996256/)

## License
Copyright (c) 2025 Tomer Pinto & Tomer Amon. All Rights Reserved.

The code in this repository is provided for demonstration and portfolio purposes only. You may view the code and run the application for personal, non-commercial use. However, you may not copy, modify, distribute, or use any part of this code in other projects, whether commercial or non-commercial, without the express written permission of the authors.