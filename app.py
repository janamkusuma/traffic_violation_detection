import streamlit as st
import cv2
from ultralytics import YOLO
import subprocess
import os
import tempfile
import shutil
import base64
import time
import pandas as pd


st.set_page_config(
    page_title="🚦 Red Light Violation Detection",
    page_icon="🚦",
    layout="wide",
)

# ---------------- Sidebar ------------------
with st.sidebar:
    st.title("🔧 Control Panel")
    st.markdown("### 📄 Instructions")
    st.markdown("""
    1. *Upload Video:* Select a traffic video file  
    2. *Process:* Click '🔍 Detect Violations'  
    3. *Download:* Get processed video & report  
    4. *Note:* Detects violations during *RED LIGHT* only
    """)
    st.markdown("### 🧬 Detection Logic")
    st.markdown("""
    *Red Light Detection:*  
    - HSV color space  
    - Bright red regions  
    - 5+ frames for stability  

    *Violation Detection:*  
    - Active only during RED LIGHT  
    - Predefined zone tracking  
    - Vehicle center points
    """)
    st.success("✅ YOLOv8 Model Loaded")
    st.markdown("### 🎯 Detection Info")
    st.markdown("""
    *Vehicle Types:* 🚗 Cars 🚛 Trucks 🚌 Buses 🏍 Motorcycles  
    *Features:*  
    - Red light state detection  
    - Violation zone monitoring  
    - Annotated video output  
    - Detailed CSV reports
    """)

# ---------------- Model ---------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

# ---------------- Utils ---------------------
def is_red_light_on(frame, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,70,50), (10,255,255))
    mask2 = cv2.inRange(hsv, (170,70,50), (180,255,255))
    return cv2.countNonZero(mask1 | mask2) > 50

def convert_avi_to_mp4(input_avi, output_mp4):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, '-y', '-i', input_avi, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', output_mp4])
        return True
    return False

def calculate_speed(prev, curr, fps, ppm=20):
    if not prev or not curr: return 0
    d = ((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)**0.5
    m = d / ppm
    return max(0, m * fps * 3.6)

# --------------- Detection ------------------
def detect_violations(video_path, model, bar):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("outputs", exist_ok=True)
    avi_out = "outputs/output.avi"
    out = cv2.VideoWriter(avi_out, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    frame_no = 0
    tracker = {}
    violations = {}
    line_y = int(h * 0.6)

    report = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1
        bar.progress(frame_no/frames)

        results = model(frame)[0]
        boxes, names = results.boxes, results.names

        red_on = any(is_red_light_on(frame, b) for b in boxes if names[int(b.cls[0])] == 'traffic light')

        current = {}
        for i, b in enumerate(boxes):
            cls = int(b.cls[0])
            if names[cls] in ['car','truck','bus','motorcycle']:
                x1,y1,x2,y2 = map(int,b.xyxy[0])
                cx,cy = (x1+x2)//2,(y1+y2)//2
                vid = f"{cls}_{i}"
                current[vid]=(cx,cy)
                spd = calculate_speed(tracker.get(vid),(cx,cy),fps)

                if red_on and cy > line_y:
                    if vid not in violations:
                        violations[vid] = {'frame':frame_no, 'cx':cx, 'cy':cy, 'speed':round(spd,2)}
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(frame,'VIOLATION!',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        tracker = current
        cv2.putText(frame,f'Total Violations: {len(violations)}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        out.write(frame)

    cap.release()
    out.release()

    mp4_out = "outputs/output.mp4"
    convert_avi_to_mp4(avi_out, mp4_out)

    for vid, data in violations.items():
        report.append({'Vehicle_ID': vid, **data})
    df_report = pd.DataFrame(report)

    return mp4_out, df_report, len(violations), frames

# ---------------- Video Display --------------
def center_video(path, width=720, height=480):
    with open(path,"rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        return f"""
        <div style='display:flex; justify-content:center;'>
            <video controls width="{width}" height="{height}">
                <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            </video>
        </div>
        """, b64

# ---------------- Main App -------------------
def main():
    st.markdown(
        """
        <div style='background-color:#e74c3c;padding:20px;border-radius:10px; text-align:center;'>
            <h1 style='color:white;'>🚦 Red Light Violation Detection System</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    model = load_model()
    uploaded = st.file_uploader("Upload traffic video", type=['mp4','avi','mov','mkv'])

    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp.write(uploaded.read())

        html_in, _ = center_video(tmp.name)
        st.markdown("### 📹 Original Video")
        st.markdown(html_in, unsafe_allow_html=True)

        if st.button("🔍 Detect Violations"):
            start = time.time()
            bar = st.progress(0)
            video_out, df_report, count, total_frames = detect_violations(tmp.name, model, bar)
            end = time.time()
            duration = round(end - start, 2)

            st.success(f"✅ Processed — Total unique violations: {count}")

            # ----- Dashboard -----
            st.markdown("## 📊 Detection Dashboard")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎞️ Total Frames", total_frames)
            col2.metric("⚡ Processing Time (s)", duration)
            col3.metric("🚦 Violations Detected", count)
            col4.metric("📄 Report Entries", df_report.shape[0])

            html_out, b64 = center_video(video_out)
            st.markdown("### 📹 Processed Video")
            st.markdown(html_out, unsafe_allow_html=True)

            st.download_button(
                "⬇ Download Processed Video",
                data=base64.b64decode(b64),
                file_name=f"violations_{int(time.time())}.mp4",
                mime="video/mp4"
            )

            st.markdown("### 📑 Violation Report")
            st.dataframe(df_report)

            csv = df_report.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download CSV Report",
                data=csv,
                file_name=f"violation_report_{int(time.time())}.csv",
                mime="text/csv"
            )

    st.markdown("---")
    st.markdown("## 💡 Tips for Best Results")
    st.markdown("""
    - Clear view of traffic lights  
    - Good lighting  
    - Show intersection clearly  
    - Visible red lights  
    - Avoid extreme weather conditions
    """)
    st.markdown("""
    <div style='text-align: center;'>
        🚦 <strong>Red Light Violation Detection</strong> | Built with Streamlit & YOLOv8<br>
        ⚡ Violations detected ONLY during RED LIGHT periods
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
