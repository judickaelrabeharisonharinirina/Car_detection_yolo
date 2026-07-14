import cv2
import streamlit as st
import requests
from ultralytics import YOLO
import time

# ==============================
# CONFIGURATION STREAMLIT
# ==============================
st.set_page_config(
    page_title="YOLOv8 Vehicle Tracking",
    layout="wide"
)

st.title("🚗 Détection & Tracking de Véhicules en Temps Réel")
st.subheader("YOLOv8 + ByteTrack + IP Webcam")

# ==============================
# CHARGEMENT MODELE
# ==============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# ==============================
# SESSION STATE
# ==============================
if "running" not in st.session_state:
    st.session_state.running = False


# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("🛠 Configuration")


source_type = st.sidebar.radio(
    "Source vidéo",
    [
        "Webcam PC",
        "Caméra téléphone IP Webcam",
        "Vidéo"
    ]
)


conf_threshold = st.sidebar.slider(
    "Confidence",
    0.1,
    1.0,
    0.5,
    0.05
)


video_source = None


# ==============================
# WEBCAM PC
# ==============================
if source_type == "Webcam PC":

    video_source = 0


# ==============================
# IP WEBCAM TELEPHONE
# ==============================
elif source_type == "Caméra téléphone IP Webcam":

    phone_ip = st.sidebar.text_input(
        "Adresse IP téléphone",
        value="192.168.88.14"
    )


    stream_path = st.sidebar.selectbox(
        "Flux vidéo",
        [
            "/video",
            "/videofeed",
            "/video.mjpg"
        ]
    )


    video_source = (
        f"http://{phone_ip}:8080{stream_path}"
    )


    st.sidebar.info(
        f"Flux :\n{video_source}"
    )


# ==============================
# VIDEO UPLOAD
# ==============================
else:

    uploaded = st.sidebar.file_uploader(
        "Choisir une vidéo",
        type=["mp4", "avi", "mov"]
    )


    if uploaded:

        filename = "temp_video.mp4"

        with open(filename, "wb") as f:
            f.write(uploaded.read())

        video_source = filename



# ==============================
# BOUTONS
# ==============================
col1, col2 = st.sidebar.columns(2)


with col1:
    if st.button("▶ Démarrer"):
        st.session_state.running = True


with col2:
    if st.button("⏹ Stop"):
        st.session_state.running = False



# ==============================
# ZONES AFFICHAGE
# ==============================
video_area = st.empty()

stats_area = st.empty()

counter_area = st.empty()



# ==============================
# TEST IP WEBCAM
# ==============================
def test_camera(url):

    try:

        r = requests.get(
            url,
            timeout=3
        )

        return True

    except:

        return False



# ==============================
# TRACKING
# ==============================
if st.session_state.running and video_source is not None:


    if source_type == "Caméra téléphone IP Webcam":

        if not test_camera(video_source):

            st.error(
                "❌ Impossible de joindre IP Webcam\n"
                "Vérifie IP + WiFi + serveur IP Webcam"
            )

            st.stop()



    cap = cv2.VideoCapture(video_source)


    if not cap.isOpened():

        st.error(
            "❌ Impossible d'ouvrir la caméra"
        )

        st.stop()



    tracked_ids = {}


    while (
        cap.isOpened()
        and st.session_state.running
    ):


        ret, frame = cap.read()


        if not ret:

            st.error(
                "❌ Impossible de lire le flux vidéo"
            )

            break



        # ======================
        # YOLO TRACKING
        # ======================

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            device=0
        )


        current_counts = {}

        annotated = frame



        for r in results:


            annotated = r.plot()


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


                for obj_id, cls in zip(ids, classes):


                    name = model.names[cls]


                    current_counts[name] = (
                        current_counts.get(name,0)+1
                    )


                    if name not in tracked_ids:

                        tracked_ids[name] = set()


                    tracked_ids[name].add(obj_id)



        # ======================
        # AFFICHAGE VIDEO
        # ======================

        video_area.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )



        # ======================
        # STATS
        # ======================

        with stats_area.container():

            st.write(
                "### 📊 Objets visibles"
            )


            if current_counts:

                for k,v in current_counts.items():

                    st.metric(
                        k.upper(),
                        v
                    )

            else:

                st.write(
                    "Aucun objet détecté"
                )



        # ======================
        # COMPTEUR GLOBAL
        # ======================

        with counter_area.container():

            st.write(
                "### 📈 Total unique"
            )


            for k,v in tracked_ids.items():

                st.success(
                    f"{k.upper()} : {len(v)}"
                )



        time.sleep(0.03)



    cap.release()
