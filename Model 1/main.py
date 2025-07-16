import numpy as np
import io
from background_removal import remove_background_with_box   
from PIL import Image
import streamlit_drawable_canvas as dc
import streamlit as st
import base64


st.set_page_config(page_title="Background Removal", layout="wide")

st.title("Team Amber: Background Removal using Logistic Regression using on-the-fly model training")

col1, col2 = st.columns(2)

with col1:
    st.header("Draw Bounding Boxes after uploading an image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        # st.image(image_np, caption="Uploaded Image", use_column_width=True)
        img_b64 = base64.b64encode(buffered.getvalue()).decode()  
        background_image_url = f"data:image/png;base64,{img_b64}"
        
        canvas_result = dc.st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=image,
            update_streamlit=True,
            height=image.size[1],
            width=image.size[0],
            drawing_mode="rect",
            display_toolbar=True,
            key="canvas",
        )
        
        # Slider for morphology kernel size
        morph_kernel_size = st.slider(
            "Morphology Kernel Size", 
            min_value=1, max_value=20, 
            value=5, 
            step=1,
            help="Adjust the size of the morphological kernel used for post-processing."
        )
        # Slider for maximum hp tuning iterations
        max_iter = st.slider(
            "Maximum Hyperparameter Tuning Iterations", 
            min_value=10, 
            max_value=100, 
            value=50, 
            step=2,
            help="Set the maximum number of iterations for hyperparameter tuning."
        )
        
        # Collect bounding boxes
        bboxes = []
        if canvas_result.json_data is not None:
            for shape in canvas_result.json_data["objects"]:
                if shape["type"] == "rect":
                    x1 = int(shape["left"])
                    y1 = int(shape["top"])
                    x2 = x1 + int(shape["width"])
                    y2 = y1 + int(shape["height"])
                    bboxes.append((x1, y1, x2, y2))
        
        if st.button("Remove Background") and bboxes:
            with st.spinner("Removing background..."):
                st.session_state.iteration = 0
                st.session_state.best_accuracy = 0.0
                mask = remove_background_with_box(
                    img=image_np, 
                    bboxes=bboxes, 
                    max_samples=10000, 
                    morph_kernel_size=morph_kernel_size, 
                    max_hyp_tuning_iter=max_iter,
                    features=['lbp', 'quest', 'hog']
                )
                
                masked_image = image_np.copy()
                masked_image[mask == 0] = [0, 0, 0]  
                
                with col2:
                    st.header("Masked Image")
                    st.image(masked_image, caption="Background removed Image", use_container_width=True)