import cv2
import os
import time
import random  
import requests
from ultralytics import YOLO

# Configuration
BACKEND_URL = "http://localhost:5000/api/violations"  

# Flow direction: "down" means normal traffic moves down the Y-axis. 
EXPECTED_FLOW_DIRECTION = "down"  

def extract_license_plate_stub(image_roi):
    """
    Mock OCR function for edge testing. 
    Generates a structured string to emulate expected LPR output.
    """
    state_code = "KA"
    district_code = f"{random.randint(1, 99):02d}"
    series = f"{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
    unique_id = f"{random.randint(1000, 9999)}"
    return f"{state_code}-{district_code}-{series}-{unique_id}"

def send_violation_to_backend(track_id, vehicle_type, violation_type, plate_number, timestamp):
    """POST violation metadata to the central API."""
    payload = {
        "vehicleId": str(track_id),
        "vehicleType": vehicle_type,
        "violationType": violation_type,
        "licensePlate": plate_number, 
        "timestamp": timestamp,
        "gpsCoordinates": "12.9716, 77.5946" 
    }
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=1)
        if response.status_code in [200, 201]:
            print(f"INFO: Synced violation for ID {track_id} ({plate_number})")
    except requests.exceptions.RequestException:
        print(f"WARNING: Network offline. Violation saved locally for ID {track_id}")

print("Initializing SurveilAI Edge Node...")

# Load model and define classes
model = YOLO('yolo11n.pt') 
TARGET_CLASSES = [2, 3, 5, 7]  # Car, motorcycle, bus, truck
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

video_path = "input1.mp4"
cap = cv2.VideoCapture(video_path)

# Spatial geometry
STOP_LINE_Y = 320  
LINE_START = (40, STOP_LINE_Y) 
LINE_END = (600, STOP_LINE_Y)

# State tracking
penalized_vehicles = set()
previous_y_positions = {}
trajectory_history = {} 

system_start_time = time.time()
print("Tracking active. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    timestamp = int(time.time())
    
    # Simulate traffic light cycle
    elapsed_time = int(time.time() - system_start_time)
    cycle = elapsed_time % 20  
    
    if cycle < 10:
        line_state, line_color = "GREEN", (0, 255, 0)
    elif cycle < 13:
        line_state, line_color = "YELLOW", (0, 255, 255)
    else:
        line_state, line_color = "RED", (0, 0, 255)
        
    cv2.line(frame, LINE_START, LINE_END, line_color, 3)
    cv2.putText(frame, f"SIGNAL: {line_state}", (LINE_START[0], LINE_START[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)
    
    # Run tracking
    results = model.track(frame, persist=True, classes=TARGET_CLASSES, conf=0.35, verbose=False)
    current_frame_ids = set()
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy() 
        
        for box, cls_id, track_id, det_conf in zip(boxes, clss, ids, confs):
            current_frame_ids.add(track_id)
            x1, y1, x2, y2 = map(int, box)
            class_name = VEHICLE_CLASSES.get(cls_id, "vehicle")
            
            bottom_center_x = int((x1 + x2) / 2)
            bottom_center_y = y2
            
            prev_y = previous_y_positions.get(track_id, None)
            previous_y_positions[track_id] = bottom_center_y  
            
            is_violating = False
            violation_type = ""
            
            # 1. Red-Light Violation
            if prev_y is not None:
                if prev_y <= STOP_LINE_Y and bottom_center_y > STOP_LINE_Y:
                    if line_state == "RED" and track_id not in penalized_vehicles:
                        penalized_vehicles.add(track_id)
                        violation_type = "Red-Light Running"
                        is_violating = True
            
            # 2. Wrong-Side Driving Vector Check
            if prev_y is not None and track_id not in penalized_vehicles:
                y_delta = bottom_center_y - prev_y
                
                if track_id not in trajectory_history:
                    trajectory_history[track_id] = []
                
                trajectory_history[track_id].append(y_delta)
                
                if len(trajectory_history[track_id]) > 8:
                    trajectory_history[track_id].pop(0)
                    
                    is_wrong_direction = (
                        all(d < -2 for d in trajectory_history[track_id]) if EXPECTED_FLOW_DIRECTION == "down"
                        else all(d > 2 for d in trajectory_history[track_id])
                    )
                    
                    if is_wrong_direction:
                        penalized_vehicles.add(track_id)
                        violation_type = "Wrong-Side Driving"
                        is_violating = True
            
            # 3. Triple Riding (Bounding box width heuristic)
            if class_name == "motorcycle" and track_id not in penalized_vehicles:
                if (x2 - x1) > 85: 
                    penalized_vehicles.add(track_id)
                    violation_type = "Triple Riding"
                    is_violating = True
            
            # Rendering and extraction
            if track_id in penalized_vehicles:
                short_codes = {
                    "Red-Light Running": "RLV",
                    "Triple Riding": "TRX",
                    "Wrong-Side Driving": "WSD"
                }
                ui_code = short_codes.get(violation_type, "ALRT")
    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID{track_id} {ui_code}", (x1, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)    
                
                if is_violating:
                    vehicle_crop = frame[max(0, y1):min(480, y2), max(0, x1):min(640, x2)]
                    extracted_plate = extract_license_plate_stub(vehicle_crop)
                    
                    cv2.putText(frame, f"PLATE: {extracted_plate}", (x1, min(y2 + 15, 475)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    print(f"ALERT: {violation_type} | ID: {track_id} | Plate: {extracted_plate}")
                    send_violation_to_backend(track_id, class_name, violation_type, extracted_plate, timestamp)
                    cv2.imwrite(f"{OUTPUT_DIR}/violation_{track_id}_{ui_code}_{timestamp}.jpg", frame)
            else:
                cv2.circle(frame, (bottom_center_x, bottom_center_y), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID {track_id}", (bottom_center_x + 5, bottom_center_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Cleanup stale tracks
    stale_ids = [tid for tid in previous_y_positions if tid not in current_frame_ids]
    for tid in stale_ids:
        previous_y_positions.pop(tid, None)
        trajectory_history.pop(tid, None)

    cv2.imshow("SurveilAI Inference Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()