# 🧠 Amor Fidelis: Expert System for the Visually Impaired

A computer vision-based expert system that detects objects using YOLOv3 and provides descriptive information for each object. Designed to assist the visually impaired through contextual awareness.

---

## 👥 Team Members

- Sneha Chakraborty (21BAI10204)
- Simran Namdev
- Shambhavi Pandey
- Divya Agarwal
- Aaron Noronha
- Aishika Ranjan
- Anushka Joshi

---

## 🔧 Technologies Used

- Python
- OpenCV
- YOLOv3
- NumPy

---

## 🗂 Project Structure

AmorFidelis_ExpertSystem/
├── main.py
├── facts/
│ ├── chair.txt
│ ├── book.txt
│ └── ...
├── coco.names
├── yolov3.cfg
├── yolov3.weights
├── README.md
├── requirements.txt


---

## 🚀 How to Run the Project

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Download the following files:
   
--yolov3.weights
--yolov3.cfg
--coco.names

3. Run the app:

--python main.py

📚 About the Facts System
Each object detected by the model is matched to a .txt file in the facts/ folder.

Example: chair.txt → contains a short description of a chair.

This creates a minimal expert system mapping visual input to contextual information.

📌 Notes
You’ll need to place a sample image like example_image.jpg in the root directory.

The program supports offline use.



