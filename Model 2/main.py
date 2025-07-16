import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import time
import tempfile
from PIL import Image
import io
import base64

# Import the segmentation function
from segment import segment_image

# Set page configuration
st.set_page_config(
    page_title="ML Background Removal",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-text {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
    }
    .feature-card {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #000;
    }
    .comparison-table {
        width: 100%;
        text-align: center;
    }
    .comparison-table th {
        background-color: #1E88E5;
        color: white;
        padding: 10px;
    }
    .comparison-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .comparison-table td {
        padding: 8px;
    }
    .result-container {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .result-image {
        text-align: center;
        width: 48%;
    }
    .result-title {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 1.2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-weight: bold;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🖼️ ML-Based Background Removal</div>', unsafe_allow_html=True)

# Introduction
st.markdown("""
    <p></p>
            
    This application removes backgrounds from images using machine learning algorithms without neural networks.
    Each pixel is classified as either foreground or background through comprehensive feature extraction and analysis.
    Choose between two ML segmentation methods:
        MultiScale Method (HMS-SRF): A hierarchical approach that processes the image at 25%, 50%, and 100% scales, with lower resolutions guiding higher-resolution segmentation.
        SingleScale Method: Direct segmentation at original resolution using a Random Forest classifier.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sub-header">Settings</div>', unsafe_allow_html=True)
    
    # Method selection
    method = st.radio(
        "Select Segmentation Method:",
        ["MultiScale", "SingleScale"],
        help="MultiScale is generally more accurate but slower, SingleScale is faster but may be less precise."
    )
    
    # Advanced options
    st.markdown("### Advanced Options")
    show_contours = st.checkbox("Show contours on visualization", value=True)
    overlay_opacity = st.slider("Mask overlay opacity", 0.0, 1.0, 0.5, 0.1)
    
    # Morphological refinement options
    st.markdown("### Morphological Refinements")
    enable_morphology = st.checkbox("Enable morphological cleanup", value=True)
    kernel_size = st.slider("Kernel size", 3, 15, 5, 2)
    
    # Morphological operations selection
    morph_operations = st.multiselect(
        "Select operations:",
        ["Opening", "Closing", "Dilation", "Erosion"],
        default=["Opening", "Closing"]
    )
    
    # About Section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app was created by Team Amber:
    - Abhishek Kumar (220101002)
    - Adarsh Gupta (220101003)
    - Vasudha Meena (220101108)
    
    CS361 Machine Learning Course Project
    """)

# Main Content Area - Two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">Upload Image</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            input_path = tmp_file.name
        
        # Button to start processing
        process_button = st.button("Remove Background")
        
        # # Information about the method
        # st.markdown(f"""
        # <div class="feature-card">
        #     <h3>About {method} Method</h3>
        #     <p>{"The MultiScale method (HMS-SRF) processes the image at three different resolutions (25%, 50%, and 100%), creating a hierarchical segmentation approach. Lower resolution predictions guide higher resolution ones for refined segmentation." 
        #        if method == "MultiScale" else 
        #        "The SingleScale method extracts 91+ features per pixel and uses a Random Forest classifier to determine foreground/background status, analyzing color, texture, edges, spatial position, and superpixel information."}</p>
        # </div>
        # """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sub-header">Results</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None and 'process_button' in locals() and process_button:
        with st.spinner('Processing image...'):
            # Create temporary files for outputs
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as mask_file:
                mask_path = mask_file.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as vis_file:
                vis_path = vis_file.name
            
            try:
                # Call the segmentation function
                start_time = time.time()
                method_param = "multiscale" if method == "MultiScale" else "singlescale"
                segment_image(input_path, mask_path, vis_path, method=method_param)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Load raw segmentation mask
                segmentation_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                # Apply morphological operations based on user settings
                if enable_morphology:
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    
                    # Apply selected morphological operations
                    if "Opening" in morph_operations:
                        segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel)
                    
                    if "Closing" in morph_operations:
                        segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel)
                    
                    if "Dilation" in morph_operations:
                        segmentation_mask = cv2.dilate(segmentation_mask, kernel, iterations=1)
                    
                    if "Erosion" in morph_operations:
                        segmentation_mask = cv2.erode(segmentation_mask, kernel, iterations=1)
                
                # Create visualization
                original_img = cv2.imread(input_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                # Create mask visualization
                vis_img = original_img.copy()
                mask_rgb = np.zeros_like(original_img)
                mask_rgb[segmentation_mask > 0] = [0, 255, 0]  # Green for foreground
                
                # Apply overlay with user-defined opacity
                vis_img = cv2.addWeighted(mask_rgb, overlay_opacity, vis_img, 1 - overlay_opacity, 0)
                
                # Draw contours if enabled
                if show_contours:
                    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 2)
                
                # Save modified visualization
                vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(vis_path, vis_img_bgr)
                
                # Display success message
                st.markdown(f'<div class="success-msg">✅ Background removed successfully in {processing_time:.2f} seconds</div>', unsafe_allow_html=True)
                
                # Display the results side by side
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                # Display mask
                st.markdown('<div class="result-image"><div class="result-title">Segmentation Mask</div>', unsafe_allow_html=True)
                st.image(segmentation_mask, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display visualization
                st.markdown('<div class="result-image"><div class="result-title">Visualization</div>', unsafe_allow_html=True)
                st.image(vis_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download buttons for results
                col_mask, col_vis = st.columns(2)
                
                with col_mask:
                    # Convert mask to bytes
                    _, buffer = cv2.imencode('.png', segmentation_mask)
                    mask_bytes = buffer.tobytes()
                    
                    # Download button
                    st.download_button(
                        label="Download Mask",
                        data=mask_bytes,
                        file_name="segmentation_mask.png",
                        mime="image/png"
                    )
                
                with col_vis:
                    # Convert visualization to bytes
                    visualization_pil = Image.fromarray(vis_img)
                    buf = io.BytesIO()
                    visualization_pil.save(buf, format='PNG')
                    vis_bytes = buf.getvalue()
                    
                    # Download button
                    st.download_button(
                        label="Download Visualization",
                        data=vis_bytes,
                        file_name="visualization.png",
                        mime="image/png"
                    )
                
                # Create transparent background version
                st.markdown('<div class="sub-header">Transparent Background</div>', unsafe_allow_html=True)
                
                # Create RGBA image with transparent background
                rgba = cv2.cvtColor(original_img, cv2.COLOR_RGB2RGBA)
                # Set alpha channel to 0 for background pixels
                rgba[:, :, 3] = segmentation_mask
                
                # Display transparent result
                st.image(rgba, caption="Image with Transparent Background", use_container_width=True)
                
                # Download transparent result
                transparent_pil = Image.fromarray(rgba)
                buf = io.BytesIO()
                transparent_pil.save(buf, format='PNG')
                transparent_bytes = buf.getvalue()
                
                st.download_button(
                    label="Download Transparent PNG",
                    data=transparent_bytes,
                    file_name="transparent_background.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
            
            # Clean up temporary files
            try:
                os.unlink(input_path)
                os.unlink(mask_path)
                os.unlink(vis_path)
            except:
                pass
    else:
        # Placeholder when no image is processed
        st.markdown("""
        <div class="card" style="text-align: center; padding: 40px;">
            <img src="https://img.icons8.com/color/96/000000/image.png" style="opacity: 0.5;"/>
            <p style="margin-top: 10px; color: #6c757d;">Upload an image and click "Remove Background" to see results here</p>
        </div>
        """, unsafe_allow_html=True)

# Morphological Refinements Section
st.markdown('<div class="sub-header">Morphological Refinements</div>', unsafe_allow_html=True)

st.markdown("""
    <p></p>

    Morphological operations help refine the segmentation mask by smoothing boundaries, removing noise, and filling holes:</p>
        Opening: Removes small objects and noise from the foreground (erosion followed by dilation)
        Closing: Fills small holes and gaps in the foreground (dilation followed by erosion)
        Dilation: Expands the foreground, useful for connecting nearby regions
        Erosion: Shrinks the foreground, useful for removing small protrusions
    Adjust the kernel size to control the strength of these operations. A larger kernel applies more aggressive refinement.</p>
</div>

<div class="feature-card">
    <h3>Tips for Better Results</h3>
    <ul>
        <li>For detailed subjects with fine edges (like hair), use a smaller kernel size (3-5)</li>
        <li>For subjects with noise or speckling, apply Opening first</li>
        <li>For subjects with holes or gaps, apply Closing</li>
        <li>The MultiScale method typically requires less morphological refinement</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; margin-bottom: 20px; color: #6c757d;">
    <p>CS361 Machine Learning Course Project<br>
    Background Removal using ML algorithms without Neural Networks</p>
</div>
""", unsafe_allow_html=True)