# SurveilAI: Next-Generation Spatio-Temporal Edge Architecture
**Gridlock Hackathon 2.0 | Prototype Phase | Theme 3**

**Team:** Info-loop 

## 🚀 Executive Summary
SurveilAI is a decentralized, edge-native traffic enforcement framework designed to eliminate manual video review. By shifting compute to localized Edge-AI nodes, we replace heavy cloud video streaming with source-isolated infraction data. 

## ⚙️ Core Architecture
* **Edge Inference:** YOLO11 optimized for local deployment enables rapid, sub-15ms frame localization.
* **Vector Tracking:** Integrated ByteTrack engines calculate cross-frame Intersection over Union (IoU) matrices to generate continuous vehicle trajectory records.
* **Geometric Triggers:** Maps irregular multi-lane `LANE_ZONES` polygons to eliminate camera perspective skews, providing a robust framework for tracking intersection breaches and wrong-side driving.
* **Context-Expanded Edge Crops:** Evaluates two-wheelers via a vertical headroom expansion routine (`crop_y1 = max(0, y1 - 0.5 * box_height)`). This guarantees the rider's entire head, hair, or protective gear is perfectly preserved within the crop boundaries.
* **Agentic VLM Validation:** Features an asynchronous Vision-Language Model (Gemini 2.5 Flash) acting as a cognitive verification layer to analyze crops via Chain-of-Thought reasoning, filtering out complex false positives (e.g., matching bare heads vs. cultural headwear) before ledger ingestion.

## 📦 Data Payload 
SurveilAI drops the raw video stream at the source node. The pipeline generates and transmits strictly a lightweight **3KB JSON metadata payload** and the single context-expanded region-of-interest (ROI) vehicle crop to preserve cellular network bandwidth.

##
*Note:
For the purpose of this hackathon's rapid feasibility prototype, the local vision engine was built using the YOLO11n architecture. This allowed for immediate, stable integration with our custom geometric tracking logic without the need to rewrite standard bounding-box pipelines over a 48-hour period.
The rapid hackathon MVP utilizes a single-threshold coordinate baseline to validate end-to-end real-time network dispatch without local compute blocks.

However, for the proposed city-wide deployment, the architecture will migrate to YOLO26. By leveraging YOLO26's native end-to-end NMS-free architecture, we expect to eliminate post-processing overhead and achieve up to 43% faster inference on CPU-bound edge devices, which is critical for scaling across municipal CCTV networks cost-effectively.

## 🛠️ Quick Start (Local Inference)

### Prerequisites
* Python 3.9+
* Active Gemini API Key (Optional for live VLM execution mode)

### Installation
Clone the repository and install the locked production dependencies:
```bash
git clone [https://github.com/hardikagarwal23/SurveilAI.git](https://github.com/hardikagarwal23/SurveilAI.git)
cd SurveilAI
pip install -r requirements.txt
