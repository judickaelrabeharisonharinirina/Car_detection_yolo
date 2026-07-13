import cv2
import streamlit as st
from ultralytics import YOLO
import os
import time


# ======================================
# CONFIGURATION
# ======================================

st.set_page_config(
    page_title="YOLOv8 Vehicle Tracking",
    layout="wide"
)

st.title("🚗 YOLOv8 Vehicle Detection + Tracking")
st.subheader("YOLOv8 + ByteTrack + Phone Camera")


# ======================================
# YOLO MODEL
# ======================================

@st.cache_resource
def load_model():

    return YOLO("yolov8n.pt")


model = load_model()



# ======================================
# SESSION
# ======================================

if "running" not in st.session_state:

    st.session_state.running = False



# ======================================
# SIDEBAR
# ======================================

st.sidebar.header("⚙️ Configuration")


source_type = st.sidebar.radio(
    "Source vidéo",
    [
        "Webcam PC",
        "Caméra téléphone",
        "Vidéo locale"
    ]
)


confidence = st.sidebar.slider(
    "Confidence",
    0.1,
    1.0,
    0.5,
    0.05
)



video_source = None



# ======================================
# WEBCAM PC
# ======================================

if source_type == "Webcam PC":

    video_source = 0



# ======================================
# TELEPHONE
# ======================================

elif source_type == "Caméra téléphone":


    ip = st.sidebar.text_input(
        "IP téléphone",
        value="192.168.88.14"
    )


    port = st.sidebar.text_input(
        "Port",
        value="4747"
    )


    endpoint = st.sidebar.selectbox(
        "Flux",
        [
            "/video",
            "/mjpegfeed?640x480",
            "/mjpegfeed?1280x720"
        ]
    )


    video_source = (
        f"http://{ip}:{port}{endpoint}"
    )


    st.sidebar.info(
        video_source
    )



# ======================================
# VIDEO
# ======================================

else:


    uploaded = st.sidebar.file_uploader(
        "Choisir vidéo",
        type=[
            "mp4",
            "avi",
            "mov"
        ]
    )


    if uploaded:


        path = "temp_video.mp4"


        with open(path,"wb") as f:

            f.write(
                uploaded.read()
            )


        video_source = path



# ======================================
# BUTTONS
# ======================================

col1,col2 = st.sidebar.columns(2)


with col1:

    if st.button("▶ Start"):

        st.session_state.running = True



with col2:

    if st.button("⏹ Stop"):

        st.session_state.running = False



# ======================================
# DISPLAY
# ======================================

video_placeholder = st.empty()

stats_placeholder = st.empty()

global_placeholder = st.empty()



# ======================================
# START TRACKING
# ======================================

if (
    st.session_state.running
    and video_source is not None
):


    # -------------------------------
    # OPEN VIDEO
    # -------------------------------


    if source_type == "Caméra téléphone":


        os.environ[
            "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        ] = (
            "fflags;nobuffer|"
            "flags;low_delay"
        )


        cap = cv2.VideoCapture(
            video_source,
            cv2.CAP_FFMPEG
        )


        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )


    else:


        cap = cv2.VideoCapture(
            video_source
        )



    if not cap.isOpened():

        st.error(
            "❌ Impossible d'ouvrir le flux vidéo"
        )

        st.stop()



    tracked_ids = {}



    # -------------------------------
    # LOOP
    # -------------------------------

    while st.session_state.running:


        ret, frame = cap.read()


        if not ret:

            st.error(
                "❌ Impossible de lire le flux"
            )

            break



        # ---------------------------
        # YOLO TRACK
        # ---------------------------

        results = model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=confidence,

            device=0,

            verbose=False
        )



        current_counts = {}

        annotated_frame = frame



        for r in results:


            annotated_frame = r.plot()



            if (
                r.boxes is not None
                and r.boxes.id is not None
            ):


                ids = (
                    r.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )


                classes = (
                    r.boxes.cls
                    .cpu()
                    .numpy()
                    .astype(int)
                )



                for obj_id, cls in zip(
                    ids,
                    classes
                ):


                    name = model.names[cls]



                    # visible

                    current_counts[name] = (
                        current_counts.get(name,0)+1
                    )



                    # global

                    if name not in tracked_ids:

                        tracked_ids[name] = set()


                    tracked_ids[name].add(
                        obj_id
                    )



        # ---------------------------
        # VIDEO DISPLAY
        # ---------------------------

        video_placeholder.image(
            annotated_frame,
            channels="BGR",
            use_container_width=True
        )



        # ---------------------------
        # CURRENT STATS
        # ---------------------------

        with stats_placeholder.container():

            st.subheader(
                "📊 Objets visibles"
            )


            if current_counts:

                for k,v in current_counts.items():

                    st.metric(
                        k.upper(),
                        v
                    )


            else:

                st.write(
                    "Aucun objet"
                )



        # ---------------------------
        # GLOBAL COUNT
        # ---------------------------

        with global_placeholder.container():

            st.subheader(
                "📈 Total unique"
            )


            for k,v in tracked_ids.items():

                st.success(
                    f"{k.upper()} : {len(v)}"
                )



        time.sleep(0.02)



    cap.release()