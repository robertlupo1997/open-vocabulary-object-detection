"""
OVOD Streamlit Demo: Interactive Open-Vocabulary Object Detection
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import io
import base64
import time
import json

# OVOD imports
import torch
from ovod.pipeline import OVODPipeline
from src.visualize import create_detection_visualization

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Cached pipeline loader - loads once and reuses"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = OVODPipeline(device=device)
    if hasattr(pipe, "load_model"):
        pipe.load_model()
    return pipe

st.set_page_config(
    page_title="OVOD Demo", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .detection-box {
        border: 2px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        background: #f8fff8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    """Load OVOD pipeline with error handling"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = OVODPipeline(device=device)
        
        # Ensure weights are loaded before UI uses .predict()
        if hasattr(pipeline, "load_model"):
            pipeline.load_model()
            
        return pipeline, None
        
    except Exception as e:
        return None, f"{e.__class__.__name__}: {e}"

# Visualization function imported from src.visualize

def display_metrics(results: dict):
    """Display performance metrics in a nice layout"""
    
    timings = results.get("timings", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Time", 
            value=f"{timings.get('total_ms', 0):.1f} ms",
            help="End-to-end pipeline latency"
        )
    
    with col2:
        st.metric(
            label="Detection", 
            value=f"{timings.get('detection_ms', 0):.1f} ms",
            help="Grounding DINO inference time"
        )
    
    with col3:
        st.metric(
            label="Segmentation", 
            value=f"{timings.get('segmentation_ms', 0):.1f} ms", 
            help="SAM 2 mask generation time"
        )
    
    with col4:
        st.metric(
            label="Objects Found", 
            value=len(results.get("boxes", [])),
            help="Number of detected objects"
        )

def create_download_button(image: Image.Image, filename: str = "ovod_result.png"):
    """Create download button for result image"""
    
    # Convert PIL image to bytes
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    
    # Create download button
    st.download_button(
        label="📥 Download Result",
        data=img_bytes,
        file_name=filename,
        mime="image/png",
        help="Download the detection result image"
    )

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 OVOD - Open Vocabulary Object Detection</h1>
        <p>Find and segment any object using natural language descriptions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Device selection
        use_gpu = st.checkbox("Use GPU (CUDA)", value=True, help="Enable GPU acceleration")
        st.session_state["use_gpu"] = use_gpu
        
        # Detection parameters
        st.subheader("Detection Settings")
        box_threshold = st.slider("Box Threshold", 0.1, 0.9, 0.35, 0.05)
        text_threshold = st.slider("Text Threshold", 0.1, 0.9, 0.25, 0.05) 
        nms_threshold = st.slider("NMS Threshold", 0.1, 0.9, 0.5, 0.05)
        max_detections = st.slider("Max Detections", 1, 100, 50, 5)
        
        # Visualization options
        st.subheader("Visualization")
        show_masks = st.checkbox("Show Segmentation Masks", value=True)
        show_scores = st.checkbox("Show Confidence Scores", value=True)
        
        # Example prompts
        st.subheader("💡 Example Prompts")
        example_prompts = [
            "person, car, dog",
            "red car and blue truck", 
            "people wearing masks",
            "construction worker with helmet",
            "laptop, phone, coffee cup",
            "cat sitting on chair"
        ]
        
        for prompt in example_prompts:
            if st.button(f"🔍 {prompt}", key=f"example_{prompt}"):
                st.session_state["prompt"] = prompt
    
    # Load pipeline
    with st.spinner("Loading OVOD pipeline..."):
        try:
            pipeline = get_pipeline()
            error = None
        except Exception as e:
            pipeline = None
            error = f"{e.__class__.__name__}: {e}"
    
    if error:
        st.error("❌ Pipeline Loading Failed")
        st.code(error)
        st.info("""
        **Troubleshooting:**
        1. Run `python demo_setup.py` to download weights
        2. Install dependencies: `pip install -r requirements.txt`
        3. Check CUDA installation if using GPU
        4. Try CPU mode in sidebar
        """)
        return
    
    st.success("✅ Pipeline loaded successfully!")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input")
        
        # Text prompt input
        prompt = st.text_input(
            "What objects to find:",
            value=st.session_state.get("prompt", ""),
            placeholder="person, car, dog...",
            help="Describe the objects you want to detect using natural language"
        )
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        # Example images
        if st.button("🖼️ Use Example Image"):
            # Create a simple example image (placeholder)
            example_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            st.session_state["example_image"] = example_img
            st.info("Using generated example image. Upload your own for better results!")
        
        # Display input image
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Input Image", use_container_width=True)
            input_image = np.array(image)
            
        elif "example_image" in st.session_state:
            input_image = st.session_state["example_image"]
            st.image(input_image, caption="📷 Example Image", use_container_width=True)
            
        else:
            st.info("👆 Upload an image or use an example image to get started")
            input_image = None
    
    with col2:
        st.header("📥 Results")
        
        if input_image is not None and prompt:
            
            if st.button("🚀 Run Detection", type="primary", use_container_width=True):
                
                # Update pipeline parameters
                pipeline.box_threshold = box_threshold
                pipeline.text_threshold = text_threshold  
                pipeline.nms_threshold = nms_threshold
                
                with st.spinner("🔍 Detecting objects..."):
                    start_time = time.time()
                    
                    # Run detection
                    results = pipeline.predict(
                        input_image,
                        prompt,
                        return_masks=show_masks,
                        max_detections=max_detections
                    )
                    
                    detection_time = time.time() - start_time
                
                # Check for errors
                if "error" in results:
                    st.error(f"❌ Detection failed: {results['error']}")
                    return
                
                # Display metrics
                st.subheader("📊 Performance Metrics")
                display_metrics(results)
                
                # Show detection results
                if len(results["boxes"]) > 0:
                    st.subheader("🎯 Detections")
                    
                    # Create visualization
                    result_image = create_detection_visualization(
                        input_image, results, show_masks
                    )
                    
                    st.image(result_image, caption="🎯 Detection Results", use_container_width=True)
                    
                    # Download button
                    create_download_button(result_image)
                    
                    # Detection details
                    with st.expander("📋 Detection Details", expanded=False):
                        for i, (label, score, box) in enumerate(zip(
                            results["labels"], results["scores"], results["boxes"]
                        )):
                            st.write(f"**{i+1}. {label}** - Confidence: {score:.3f}")
                            if show_scores:
                                st.write(f"   📍 Box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
                    
                    # Raw results (for debugging)
                    with st.expander("🔧 Raw Results (Debug)", expanded=False):
                        st.json({
                            "prompt": results["prompt"],
                            "grounding_prompt": results["grounding_prompt"], 
                            "object_list": results["object_list"],
                            "num_detections": len(results["boxes"]),
                            "timings": results["timings"],
                            "image_shape": results["image_shape"]
                        })
                
                else:
                    st.warning("🤷 No objects found. Try:")
                    st.write("• Lowering the detection threshold")
                    st.write("• Using more specific object descriptions")  
                    st.write("• Checking if objects are clearly visible")
        
        elif not prompt:
            st.info("💬 Enter a text prompt describing what to find")
        else:
            st.info("🖼️ Upload an image to analyze")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        Built with ❤️ using Grounding DINO + SAM 2 | 
        <a href="https://github.com/IDEA-Research/GroundingDINO">Grounding DINO</a> | 
        <a href="https://github.com/facebookresearch/segment-anything-2">SAM 2</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
