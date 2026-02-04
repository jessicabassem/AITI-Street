# AiTi Street  
### AI-Based Smart Traffic Management & Pollution Reduction System

## 📌 Overview
AiTi Street is a smart traffic control system designed to reduce traffic congestion, emergency response time, and air pollution in urban areas.  
The system was developed as a prototype simulating **20th Street, Boulaq Al-Dakror, Cairo**, integrating **AI, IoT, and material science** solutions.

Our approach combines:
- AI-controlled adaptive traffic lights
- Emergency vehicle prioritization (green corridors)
- Real-time traffic detection using computer vision
- Siren detection using sound classification
- Photocatalytic road materials for NOx reduction
- A mobile application for route and weather guidance

---

## 🎯 Problem Statement
Traffic congestion in Egypt causes:
- Severe delays and economic loss
- Increased NOx and PM2.5 pollution
- Higher accident rates
- Delayed emergency vehicle response

AiTi Street targets these issues by addressing **both the root cause (traffic flow)** and **its consequences (pollution and safety)**.

---

## 🧠 System Architecture
The system operates through synchronized modules:

- **YOLOE-11S AI Model** for vehicle detection
- **YAMNet** for emergency siren detection
- **Arduino-controlled traffic lights**
- **Green corridor activation** for emergency vehicles
- **Flutter mobile application**
- **N-doped TiO₂ photocatalytic cement** for pollution reduction

---

## 🚦 Features
- Adaptive traffic light timing based on real-time congestion  
- Emergency vehicle prioritization with response time of **2.26 seconds**  
- AI accuracy up to **98.7%**  
- Trip time reduction by **34.45%**  
- NO pollution reduction up to **79%**  
- Weather-aware route recommendation via mobile app  

---

## 🛠️ Technologies Used
- **AI & ML:** YOLOE-11S, YAMNet  
- **Hardware:** Arduino, DC motors, sensors  
- **Mobile App:** Flutter  
- **Materials Science:** N-doped TiO₂ photocatalytic cement  
- **Dataset Tools:** Roboflow, data augmentation  

---

## 🧪 Dataset & Training
- Initial dataset: **2,878 images**
- After augmentation: **6,926 images**
- Split:
  - Training: 6,072
  - Validation: 571
  - Testing: 283

Augmentation techniques included:
- Motion blur
- Zoom
- Shear
- Multi-exposure simulation

---

## 📊 Results
| Metric | Result |
|------|-------|
| AI Accuracy | 98.7% |
| Response Time | 2.26 s |
| Trip Time Reduction | 34.45% |
| NO Reduction | 79% |

---

## 🖼️ Prototype & System Images

### Traffic Simulation Model
![Traffic Model](images/traffic_model.jpg)

### AI Vehicle Detection
![YOLO Detection](images/yolo_detection.jpg)

### Emergency Vehicle Detection
![Emergency Detection](images/emergency_detection.jpg)

### Mobile Application
![App Interface](images/app_interface.jpg)

### Photocatalytic Cement Preparation
![TiO2 Preparation](images/tio2_preparation.jpg)


---

## 🚀 Future Improvements
- Replace YOLOE with Vision Transformers (ViT) for large-scale deployment  
- Use higher-quality microphones for siren detection  
- Increase N-doping efficiency using Melamine  
- Deploy system at city scale  

