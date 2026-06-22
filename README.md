# SurveilAI: Next-Generation Spatio-Temporal Edge Architecture
**Gridlock Hackathon 2.0 | Prototype Phase | Theme 3**

**Team:** Info-loop 

## 🚀 Executive Summary
SurveilAI is a decentralized, edge-native traffic enforcement framework designed to eliminate manual video review. By shifting compute to localized Edge-AI nodes, we replace heavy cloud video streaming with source-isolated infraction data. 

## ⚙️ Core Architecture
* **Edge Inference:** YOLO11 (via TensorRT) enables sub-15ms detection. 
* **Vector Tracking:** ByteTrack calculates cross-frame IoU for continuous vehicle trajectory vectors.
* **Geometric Triggers:** Deterministic polygons map multi-lane `LANE_ZONES` to instantly flag Wrong-Side Driving and Red-Light Violations.
* **Cascaded Two-Wheeler Pipeline:** Secondary pose-estimation isolates Two-Wheeler bounding boxes to detect Triple Riding and Helmet Non-Compliance.
* **VLM Validation:** An Agentic Vision-Language Model acts as a cloud-barrier to reject culturally specific false positives (e.g., turbans flagged as no-helmet) before logging.

## 📦 Data Payload 
SurveilAI drops the video feed at the edge. It transmits strictly a **3KB JSON metadata payload** and a single highly-compressed ROI image crop to the Kafka Data Broker.

##
*For the purpose of this hackathon's rapid feasibility prototype, the local vision engine was built using the YOLO11n architecture. This allowed for immediate, stable integration with our custom geometric tracking logic without the need to rewrite standard bounding-box pipelines over a 48-hour period.

However, for the proposed city-wide deployment, the architecture will migrate to YOLO26. By leveraging YOLO26's native end-to-end NMS-free architecture, we expect to eliminate post-processing overhead and achieve up to 43% faster inference on CPU-bound edge devices, which is critical for scaling across municipal CCTV networks cost-effectively.

## 🛠️ Quick Start (Local Inference)

### Prerequisites
* Python 3.9+
* Recommended: Nvidia GPU with CUDA enabled

### Installation
Clone the repository and install the strictly required dependencies:
```bash
git clone [https://github.com/hardikagarwal23/SurveilAI.git](https://github.com/hardikagarwal23/SurveilAI.git)
cd SurveilAI
pip install -r requirements.txt
