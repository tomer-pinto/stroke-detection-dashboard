# stroke_detection_dashboard.py

import streamlit as st
import tensorflow as tf
import cv2
import albumentations as A
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score

LOGO = "https://i.postimg.cc/kXbvWYpH/logo.png"

# --- Page Configuration ---
st.set_page_config(
    page_title="Stroke Detection Dashboard",
    page_icon=LOGO,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
            display: none !important;
            visibility: hidden !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Initialize Session State ---
if 'view' not in st.session_state:
    st.session_state.view = 'menu'
if 'file_to_process' not in st.session_state:
    st.session_state.file_to_process = None

# --- Helper & Core Functions ---
@st.cache_resource
def load_app_data():
    """
    Load all required assets: config, face detector, and ML models.
    This function is cached to run only once.
    """
    assets = {}
    CONFIG_PATH = "config.json"
    PROTOTXT_PATH = "files/deploy.prototxt"
    CAFFEMODEL_PATH = "files/res10_300x300_ssd.caffemodel"

    try:
        with open(CONFIG_PATH, 'r') as f:
            assets['config'] = json.load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {CONFIG_PATH}")
        return None

    # Determine top 3 models for the ensemble
    model_scores = assets['config']['model_scores']
    sorted_models = sorted(model_scores.items(), key=lambda item: item[1], reverse=True)
    assets['top_3_names'] = [name for name, _ in sorted_models[:3]]

    # Load all ML Models specified in the config
    assets['ml_models'] = {}
    for name in model_scores.keys():
        path = f'models/best_model_{name}.keras'
        if not os.path.exists(path):
            st.error(f"Missing model file: {path}")
            return None
        try:
            assets['ml_models'][name] = tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            st.error(f"Error loading model {path}: {e}")
            return None

    # Load the OpenCV DNN Face Detector
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFEMODEL_PATH):
        st.error("Face detector model files not found.")
        return None
    assets['face_detector'] = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

    return assets

# Defines the standard image transformations for the model
val_test_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

def predict(image_bytes, assets, model_choice):
    """Run the full prediction pipeline: detect face, preprocess, and predict."""
    try:
        image = Image.open(image_bytes).convert('RGB')
        image_np = np.array(image)
        face_net = assets['face_detector']

        (h, w) = image_np.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        # Find the largest face in the image
        best_box, max_area = None, 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area, best_box = area, box.astype("int")

        if best_box is None: 
            return "Error: No face detected in image.", None, None

        # Crop the face with padding
        (startX, startY, endX, endY) = best_box
        padding = 0.1
        pad_w, pad_h = int((endX - startX) * padding), int((endY - startY) * padding)
        startX, startY = max(0, startX - pad_w), max(0, startY - pad_h)
        endX, endY = min(w-1, endX + pad_w), min(h-1, endY + pad_h)
        face = image_np[startY:endY, startX:endX]

        # Preprocess the cropped face for the model
        transformed = val_test_transforms(image=face)
        image_tensor = tf.convert_to_tensor(transformed['image'], dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        # Get the prediction score from the selected model
        threshold = assets['config']['thresholds'][model_choice]
        if model_choice == "Ensemble":
            preds = [assets['ml_models'][name].predict(image_tensor, verbose=0) for name in assets['top_3_names']]
            raw_prediction = np.mean(preds)
        else:
            best_model_name = assets['top_3_names'][0]
            raw_prediction = assets['ml_models'][best_model_name].predict(image_tensor, verbose=0)[0][0]

        prediction = "Suspected Stroke" if raw_prediction > threshold else "No Stroke Suspected"
        return prediction, raw_prediction, face
    except Exception as e:
        return f"An error occurred during analysis: {e}", None, None

@st.dialog(" ")
def show_emergency_dialog():
    """
    Displays a prominent, well-styled emergency alert that includes
    the F.A.S.T. acronym and the emergency number.
    """
    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style='color: white;'>🚨 Suspected Medical Emergency! 🚨</h2>
            <h3 style='color: #FF4B4B;'>Calling Emergency Services...</h3>
            <div style="text-align: left; background-color: #262730; padding: 15px; border-radius: 10px; margin-top: 20px;">
                <h4 style="margin-bottom: 10px;">Remember the signs of a stroke (F.A.S.T.):</h4>
                <ul>
                    <li><strong>Face:</strong> Is one side of the face drooping?</li>
                    <li><strong>Arms:</strong> Is one arm weak or numb?</li>
                    <li><strong>Speech:</strong> Is speech slurred or strange?</li>
                    <li><strong>Time:</strong> Time is critical - call emergency services immediately!</li>
                </ul>
            </div>
            <p style='font-size: 1.1em; margin-top: 20px; font-weight: bold;'>
                This is a demonstration only. In a real emergency, call for help immediately.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

##################################################################################################################################################################################################################################################################################################################################################################################################

# --- Plotting Helper Functios ---
def plot_confusion_matrix(cm_data, title, caption):
    """Creates and displays a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["no_stroke", "stroke"],
                yticklabels=["no_stroke", "stroke"])
    ax.set_title(title)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(caption)
    plt.close(fig)

def plot_roc_curve(y_true, y_scores, roc_auc, title="ROC Curve"):
    """Creates and displays a ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(f"The Area Under the Curve (AUC) is {roc_auc:.4f}, indicating the model's discriminative ability.")
    plt.close(fig)

def plot_pr_curve(y_true, y_scores, ap_score, title="Precision-Recall Curve"):
    """Creates and displays a Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color='purple', lw=2, label=f'AP = {ap_score:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="lower left")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(f"The Average Precision (AP) score is {ap_score:.4f}, summarizing the precision-recall trade-off.")
    plt.close(fig)

##################################################################################################################################################################################################################################################################################################################################################################################################

# --- Main App ---

# --- Loading Screen ---
if "loader_shown" not in st.session_state:
    st.session_state.loader_shown = False

loader_placeholder = st.empty()

if not st.session_state.loader_shown:
    loader_html = f"""
    <style>
        .loader-container {{
            position: fixed; top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: #121212;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            z-index: 9999;
        }}
        .loader-container img {{
            width: 300px;
            animation: pulse 1.5s infinite ease-in-out;
            border-radius: 10px;
        }}
        .loader-text {{
            margin-top: 20px; font-size: 1.1em; color: white;
            text-align: center; font-family: sans-serif;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
    </style>

    <div class="loader-container" id="loader">
        <img src="{LOGO}">
        <div class="loader-text">Loading Stroke Detection Dashboard...</div>
    </div>
    """
    loader_placeholder.markdown(loader_html, unsafe_allow_html=True)

    start_time = time.time()
    
    app_assets = load_app_data()

    elapsed_time = time.time() - start_time
    minimum_display_time = 3

    if elapsed_time < minimum_display_time:
        time.sleep(minimum_display_time - elapsed_time)

    loader_placeholder.empty()
    st.session_state.loader_shown = True
    st.rerun()

else:
    app_assets = load_app_data()

# --- Main Content ---
if app_assets:

    # --- Main Tabs Navigation ---
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Prediction", "📊 Model Insights", "ℹ️ About the Project", "🤝 Contact Us"])

###################################################################################################################################################################################################################################################################################################################################################################################################
    
    # --- TAB 1: PREDICTION TOOL ---
    with tab1:
        control_col, main_col = st.columns([0.35, 0.65])

        with control_col:
            st.header("Model Selection")
            best_name = app_assets['top_3_names'][0]
            radio_labels = {"Ensemble": "Ensemble (Recommended)", "Single Best": f"{best_name} (Single Best)"}
            model_selection = st.radio("Choose a model to run:", options=radio_labels.keys(), format_func=lambda k: radio_labels[k])
            
            with st.expander("Remember the signs of a stroke (F.A.S.T.)", expanded=True):
                st.markdown("""
                - **F - Face:** Is one side of the face drooping?
                - **A - Arms:** Is one arm weak or numb?
                - **S - Speech:** Is speech slurred or strange?
                - **T - Time:** Time is critical - **CALL EMERGENCY SERVICES IMMEDIATELY!**
                """)
        
        # --- Main Content Area ---
        with main_col:
            def set_view(view_name):
                st.session_state.view = view_name
            def on_file_change(uploader_key):
                if st.session_state[uploader_key] is not None:
                    st.session_state.file_to_process = st.session_state[uploader_key]
                    st.session_state.view = 'results'

            # View 1: Main Menu
            if st.session_state.view == 'menu':
                st.title("Stroke Detection Dashboard")
                st.markdown("This tool provides a preliminary analysis of a facial image for potential signs of a stroke.")
                st.markdown("---")
                st.subheader("Choose Input Method")
                btn_col1, btn_col2 = st.columns(2)
                btn_col1.button("📂 Upload Image", use_container_width=True, on_click=set_view, args=('upload',))
                btn_col2.button("📸 Take a Photo", use_container_width=True, on_click=set_view, args=('camera',))
            
            # View 2: Input Screens
            elif st.session_state.view in ['upload', 'camera']:
                if st.session_state.view == 'upload':
                    st.file_uploader("Select a photo", type=['jpg','jpeg','png'], key='uploader', on_change=on_file_change, args=('uploader',))
                elif st.session_state.view == 'camera':
                    st.camera_input("Center your face", key='camera', on_change=on_file_change, args=('camera',))
                st.button("Back to Menu", on_click=set_view, args=('menu',))
            
            # View 3: Results Screen
            elif st.session_state.view == 'results':
                st.subheader("Analysis Results")
                file_to_process = st.session_state.file_to_process
                if file_to_process:
                    res_col1, res_col2 = st.columns([2, 3])
                    with st.spinner('Analyzing...'):
                        prediction, score, face_img = predict(file_to_process, app_assets, model_selection)
                    
                    with res_col1:
                        if face_img is not None: 
                            st.image(face_img, use_container_width=True, caption="Detected Face")
                        else: 
                            st.warning(prediction)
                    
                    with res_col2:
                        if score is not None:
                            if prediction == "Suspected Stroke": 
                                st.error(f"**Prediction:** {prediction} 🚨")
                                show_emergency_dialog()
                            else: 
                                st.success(f"**Prediction:** {prediction} ✅")

                            st.metric(label="Model Confidence Score", value=f"{score*100:.2f}%")
                            threshold = app_assets['config']['thresholds'][model_selection]
                            st.progress(float(min(score / (threshold * 1.5), 1.0)))
                            st.caption(f"A score above {threshold*100:.2f}% is classified as 'Suspected Stroke'.")
                    
                    st.button("Analyze Another Image", on_click=set_view, args=('menu',), use_container_width=True)

##################################################################################################################################################################################################################################################################################################################################################################################################    

    # --- TAB 2: MODEL INSIGHTS ---
    with tab2:
        st.header("Model Performance Insights")
        st.markdown("This tab offers a complete analysis of the model evaluation process, from a high-level summary to an in-depth comparison and interactive tools.")

        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            "📊 Performance Summary", 
            "🔍 In-Depth Comparison", 
            "🛠️ Interactive Explorer"
        ])

##################################################################################################################################################################################################################################################################################################################################################################################################
        
        # --- Sub Tab 1: Performance Summary ---
        with sub_tab1:
            st.subheader("Executive Summary")
            st.markdown("This section presents the final, high-level results, including the key metrics of our recommended model and a comparison of all tested candidates.")
            
            # 1. KPIs for the champion model
            st.markdown("#### Key Performance Indicators (Ensemble Model)")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            kpi_col1.metric("Recall (Stroke)", "67%", help="Percentage of actual strokes correctly identified.")
            kpi_col2.metric("Precision (Stroke)", "86%", help="Of all 'stroke' predictions, this percentage was correct.")
            kpi_col3.metric("F1-Score (Stroke)", "75%", help="A balance between precision and recall.")
            kpi_col4.metric("Overall Accuracy", "88%", help="The percentage of all predictions that the model got correct.")
            
            st.markdown("---")

            # 2. Bar Chart comparing all candidates
            st.subheader("Individual Model Scores (Validation Set)")
            _, middle_col, _ = st.columns([1, 2, 1])
            with middle_col:
                scores = app_assets['config']['model_scores'].copy()
                scores['EfficientNetB0'] = 0.7727
                sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

                fig, ax = plt.subplots(figsize=(5, 4))
                colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
                bars = sns.barplot(x=list(sorted_scores.keys()), y=list(sorted_scores.values()), ax=ax, palette=colors)

                for bar, v in zip(bars.patches, sorted_scores.values()):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f"{v:.2f}",
                            ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

                ax.set_xlabel("Model Name", fontsize=9)
                ax.set_ylabel("AUC Score", fontsize=9)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                st.pyplot(fig)
                plt.close(fig)
                st.caption("This chart shows the performance of all individual models tested, justifying the selection of the top three for the ensemble.")
            
            st.markdown("---")

            # 3. Final Conclusion and Recommendation
            st.subheader("Final Recommendation")
            st.success("""
            **The Ensemble model is the recommended choice.** While both the Ensemble and the single best model (ResNet50V2) achieve a similar, high F1-Score (0.75) and AUC (~0.91), the Ensemble model provides a higher **Overall Accuracy (88% vs. 87%)** and, most importantly, a significant improvement in **Precision**. This leads to fewer false alarms, making it a more reliable and trustworthy model for this application.
            """, icon="🏆")

##################################################################################################################################################################################################################################################################################################################################################################################################

        # --- Sub Tab 2: In-Depth Model Comparison ---
        with sub_tab2:
            st.subheader("In-Depth Model Comparison")
            st.markdown("""
            This section provides a side-by-side deep dive into the performance of the best single model versus the final ensemble model. 
            All metrics are calculated live from the test set results for maximum consistency.
            """)

            try:
                y_true = np.load('ynpy/y_true.npy').ravel()
                y_scores_single = np.load('ynpy/y_scores_single.npy')
                y_scores_ensemble = np.load('ynpy/y_scores_ensemble.npy')
                threshold_single = app_assets['config']['thresholds']['Single Best']
                threshold_ensemble = app_assets['config']['thresholds']['Ensemble']

                left_model_col, right_model_col = st.columns(2)

                # --- ResNet50V2 Column ---
                with left_model_col:
                    with st.container(border=True):
                        st.markdown("<h4 style='text-align: center;'>ResNet50V2 (Best Single Model)</h4>", unsafe_allow_html=True)
                        y_pred_single = (y_scores_single >= threshold_single).astype(int)
                        cm_single = confusion_matrix(y_true, y_pred_single, labels=[0, 1])
                        plot_confusion_matrix(cm_single, f"Confusion Matrix (T={threshold_single:.4f})", f"The model missed {cm_single[1, 0]} stroke cases (False Negatives).")
                        plot_roc_curve(y_true, y_scores_single, auc(roc_curve(y_true, y_scores_single)[0], roc_curve(y_true, y_scores_single)[1]))
                        plot_pr_curve(y_true, y_scores_single, average_precision_score(y_true, y_scores_single))

                # --- Ensemble Column ---
                with right_model_col:
                    with st.container(border=True):
                        st.markdown("<h4 style='text-align: center;'>Ensemble Model</h4>", unsafe_allow_html=True)
                        y_pred_ensemble = (y_scores_ensemble >= threshold_ensemble).astype(int)
                        cm_ensemble = confusion_matrix(y_true, y_pred_ensemble, labels=[0, 1])
                        plot_confusion_matrix(cm_ensemble, f"Confusion Matrix (T={threshold_ensemble:.4f})", f"The model missed {cm_ensemble[1, 0]} stroke cases (False Negatives).")
                        plot_roc_curve(y_true, y_scores_ensemble, auc(roc_curve(y_true, y_scores_ensemble)[0], roc_curve(y_true, y_scores_ensemble)[1]))
                        plot_pr_curve(y_true, y_scores_ensemble, average_precision_score(y_true, y_scores_ensemble))
            
            except FileNotFoundError:
                st.error("Error: Make sure `y_true.npy`, `y_scores_single.npy`, and `y_scores_ensemble.npy` are in the project/ynpy folder.")

            st.markdown("---")

            # --- Final Summary and Thresholds Section ---
            st.subheader("Summary & Decision Thresholds")
            summary_col, table_col = st.columns([0.6, 0.4])
            with summary_col:
                st.info("""
                **Key Conclusions:**
                While both models are strong, as indicated by their high and nearly identical AUC scores, the **Ensemble model is the recommended choice**. The models also achieve a similar F1-Score (0.75). However, a deeper look at the trade-offs reveals the Ensemble's advantage:
                - Its key strength is superior **Precision (86% vs. 82%)**, significantly reducing the number of false alarms (False Positives).
                - This comes at the cost of a marginally lower **Recall (67% vs. 69%)**.
                For a practical application, reducing false alarms is critical for user trust. Therefore, the Ensemble's profile is more balanced and reliable.
                """, icon="💡")
            
            with table_col:
                thresholds = app_assets['config']['thresholds']
                best_name = app_assets['top_3_names'][0]
                thresholds_data = {
                    "Model": [f"Single Best ({best_name})", "Ensemble"],
                    "Optimal Threshold": [f"{thresholds['Single Best']:.4f}", f"{thresholds['Ensemble']:.4f}"]
                }
                df_thresh = pd.DataFrame(thresholds_data)
                st.dataframe(df_thresh, use_container_width=True, hide_index=True)

##################################################################################################################################################################################################################################################################################################################################################################################################

        #--- Sub Tab 3: Interactive Explorer ---
        with sub_tab3:
            st.subheader("Interactive Threshold Explorer (Ensemble Model)")
            st.markdown("This section demonstrates the trade-off between catching more stroke cases (higher Recall) and making fewer false alarms (higher Precision) as you adjust the decision threshold.")

            try:
                y_true = np.load('ynpy/y_true.npy').ravel()
                y_scores_ensemble = np.load('ynpy/y_scores_ensemble.npy')
                
                st.markdown("#### Threshold Adjustment")
                slider_col, number_col = st.columns([0.7, 0.3])
                with slider_col:
                    slider_threshold = st.slider(
                        "Move the slider:", min_value=0.0, max_value=1.0, value=0.5657, step=0.0001,
                        format="%.4f", key="threshold_slider"
                    )
                with number_col:
                    new_threshold = st.number_input(
                        "Or enter a precise value:", min_value=0.0, max_value=1.0, value=slider_threshold, step=0.0001,
                        format="%.4f", key="threshold_number_input"
                    )

                st.subheader("Model Score Distribution")
                fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                df_scores = pd.DataFrame({'score': y_scores_ensemble, 'actual_class': y_true})
                df_scores['actual_class'] = df_scores['actual_class'].map({0: 'no_stroke', 1: 'stroke'})
                sns.histplot(data=df_scores, x='score', hue='actual_class', kde=True, ax=ax_hist, palette={'no_stroke': 'skyblue', 'stroke': 'salmon'})
                ax_hist.axvline(x=new_threshold, color='green', linestyle='--', linewidth=2, label=f'Current Threshold ({new_threshold:.4f})')
                ax_hist.set_title("Distribution of Prediction Scores by Actual Class")
                ax_hist.set_xlabel("Predicted Score (Probability of Stroke)")
                ax_hist.set_ylabel("Number of Cases")
                ax_hist.legend()
                st.pyplot(fig_hist)
                plt.close(fig_hist)
                st.caption("This plot shows the distribution of the model's scores for both true classes. The overlap between the blue ('no_stroke') and salmon ('stroke') distributions represents the 'confusion zone' where the model is most likely to make errors.")
                
                st.markdown("---")

                st.subheader("Live Performance Metrics")
                y_pred_interactive = (y_scores_ensemble >= new_threshold).astype(int)
                kpi_col, cm_col = st.columns(2)
                with kpi_col:
                    precision = precision_score(y_true, y_pred_interactive, pos_label=1, zero_division=0)
                    recall = recall_score(y_true, y_pred_interactive, pos_label=1)
                    f1 = f1_score(y_true, y_pred_interactive, pos_label=1)
                    st.metric("Recall (Stroke)", f"{recall:.0%}")
                    st.metric("Precision (Stroke)", f"{precision:.0%}")
                    st.metric("F1-Score (Stroke)", f"{f1:.0%}")

                with cm_col:
                    cm_interactive = confusion_matrix(y_true, y_pred_interactive, labels=[0, 1])
                    plot_confusion_matrix(cm_interactive, f"Confusion Matrix (T={new_threshold:.4f})", f"The model missed {cm_interactive[1, 0]} stroke cases (False Negatives).")

            except FileNotFoundError:
                st.error("Error: Make sure 'y_true.npy' and 'y_scores_ensemble.npy' are in the project/ynpy folder to enable this feature.")

##################################################################################################################################################################################################################################################################################################################################################################################################

    #--- TAB 3: ABOUT THE PROJECT ---
    with tab3:
        st.header("About the Project")
        st.markdown("This dashboard was developed as a final project for a B.Sc. in Information Systems Engineering, specializing in Data Science.")

        st.subheader("🎯 Project Goal")
        st.markdown("""
        The main objective of this project is to explore the potential of using Artificial Intelligence and Deep Learning for early stroke detection. Key goals include:
        - **AI Screening:** This project explores the potential of using Artificial Intelligence and Deep Learning to create a rapid, accessible, and non-invasive screening tool. The primary objective is to develop a model capable of detecting early signs of facial asymmetry commonly associated with an ischemic stroke, by analyzing a single facial image.
        - **Early Warning:** The goal is not to replace a medical diagnosis but to provide a preliminary alert system that could encourage individuals to seek immediate medical attention, which is critical in stroke cases.           
        """)

        st.markdown("---")

        st.subheader("⚙️ Methodology")
        st.markdown("""
        The development process followed several key stages to build and evaluate the models:
        - **Dataset:** The model was trained on a challenging public dataset of facial images, which were only categorized as 'stroke' or 'no_stroke' with no additional metadata. The dataset was highly imbalanced, consisting of 3,770 heavily augmented images derived from only 68 unique individuals (47 healthy, 21 with stroke). The poor quality of the images (including blurriness, rotations, and extreme augmentations) required meticulous manual clustering by identity to prevent data leakage and train a reliable model.
        - **Model Architecture:** Several state-of-the-art, pre-trained Convolutional Neural Network (CNN) architectures were utilized, including ResNet50V2, ConvNeXtTiny, MobileNetV2, and EfficientNetB0. Each model was then fine-tuned for the specific task of stroke detection. After evaluation, the three models that yielded the best performance for this task (**ResNet50V2, ConvNeXtTiny, and MobileNetV2**) were selected for the final analysis and for inclusion in the ensemble model.
        - **Ensemble Technique:** To enhance performance and reliability, an **Ensemble model** was developed. This model aggregates the predictions from the top three individual models. By averaging their outputs, the ensemble approach reduces the risk of an error from a single model and leads to a more robust and balanced final decision.
        """)

        st.markdown("---")        

        st.subheader("📜 About This Dashboard")
        st.markdown("""
        This interactive dashboard serves as the user interface for the project's final models. It provides two main functionalities:
        1.  **Prediction Tool:** Allows users to get a live prediction by either uploading an image or using a live camera feed.
        2.  **Model Insights:** Offers a comprehensive analysis of the models' performance, including key metrics (KPIs), comparative plots, and interactive tools.
        """)

        st.markdown("---")

        st.subheader("🔗 Links & Resources")
        st.markdown("""
        For a deeper look into the project's implementation and results, please refer to the following resources:
        - **[Explore the Full Pipeline in the Colab Notebook](https://colab.research.google.com/drive/10Oa7w8Wr3CZI_n7Y1kR4jdlFCY87AfRl?usp=sharing)**
        - **[View Project on GitHub](https://github.com/tomer-pinto/stroke-detection-dashboard.git)**
        """)

        st.markdown("---")

        st.subheader("Important Disclaimer")
        st.warning("""
        **Disclaimer: This is an academic proof-of-concept and is NOT a substitute for professional medical advice.**
        
        The tool's predictions should not be used for self-diagnosis or to delay seeking medical care. The accuracy of the model is limited by the data it was trained on. If you or someone you know exhibits any signs of a stroke (F.A.S.T.), contact emergency services immediately.
        """, icon="⚠️")

##################################################################################################################################################################################################################################################################################################################################################################################################

    #--- TAB 4: CONTACT US ---
    with tab4:
        pinto_img_url = "https://i.postimg.cc/W3YwwBVB/tomer-pinto-profile.jpg"
        amon_img_url = "https://i.postimg.cc/QCRgRcHS/tomer-amon-profile.jpg"

        page_html = f"""
        <style>
            .contact-container {{
                display: flex;
                justify-content: center;
                align-items: stretch;
                gap: 40px;
                margin-top: 40px;
                flex-wrap: wrap;
            }}
            .card {{
                border: 2px solid #333;
                border-radius: 15px;
                padding: 25px;
                width: 320px;
                background-color: #1C1C1C;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                transition: all 0.3s ease-in-out;
                border-radius: 12px;
            }}
            .card img.profile-pic {{
                width: 150px;
                height: 150px;
                border-radius: 50%;
                object-fit: cover;
                margin-bottom: 15px;
            }}
            .card h3 {{
                color: white;
                margin: 0 0 0 7px;
            }}
            .buttons {{
                margin-top: 15px;
                display: flex;
                justify-content: center;
                gap: 10px;
            }}
            .btn {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 10px 18px;
                border-radius: 8px;
                font-weight: 500;
                text-decoration: none !important;
                color: white !important;
                font-size: 15px;
            }}
            .btn img.icon {{
                width: 20px;
                height: 20px;
                margin-bottom: 3px;
            }}
            .linkedin {{ background-color: #0077B5; }}
            .github {{ background-color: #333; }}
            .github img {{ filter: invert(1); }}
            .btn:hover {{ opacity: 0.85; }}
            .card:hover {{
                transform: translateY(-10px) scale(1.02);
                box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
            }}
            .footer {{
                margin-top: auto;
                text-align: center;
                padding: 10px;
                font-size: 20px;
                font-weight: bold;
            }}
        </style>

        <h2>Contact Us</h2>
        <p>If you have any questions, feedback, or inquiries about this project, please feel free to reach out to us:</p>
        <div class="contact-container">
            <div class="card">
                <img src="{pinto_img_url}" alt="" class="profile-pic">
                <h3>Tomer Pinto</h3>
                <div class="buttons">
                    <a href="https://www.linkedin.com/in/tomerpinto/" target="_blank" class="btn linkedin">
                        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" class="icon"> LinkedIn
                    </a>
                    <a href="https://github.com/tomer-pinto" target="_blank" class="btn github">
                        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" class="icon"> GitHub
                    </a>
                </div>
            </div>
            <div class="card">
                <img src="{amon_img_url}" alt="" class="profile-pic">
                <h3>Tomer Amon</h3>
                <div class="buttons">
                    <a href="https://www.linkedin.com/in/tomer-amon-9aa996256/" target="_blank" class="btn linkedin">
                        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" class="icon"> LinkedIn
                    </a>
                </div>
            </div>
        </div>
        <div class="footer">
            We appreciate your interest and support!
        </div>
        """

        st.markdown(page_html, unsafe_allow_html=True)

##################################################################################################################################################################################################################################################################################################################################################################################################        

else:
    st.error("App cannot start. Please ensure all model and configuration files are in the correct folder.")