# Pipe Detection using YOLOv8

This project detects pipes in images and live video feeds using a trained YOLOv8 model. The application is built with Flask and supports both desktop and mobile web interfaces.

## Features
- Upload an image and detect pipes.
- Live video detection with real-time object counting.
- Interactive UI for both desktop and mobile users.
- Secure HTTPS support.

## Installation
### **1. Clone the Repository**
```sh
git clone https://github.com/KrishKapadia09/PipeDetection.git
cd PipeDetection
```

### **2. Create a Virtual Environment (Optional but Recommended)**
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **3. Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4. Place YOLO Model Weights**
Download or train a YOLOv8 model and place `best.pt` in the project root directory.

### **5. Run the Flask App**
```sh
python app.py
```
The app will run on `https://127.0.0.1:5000/` by default.

## Usage
### **1. Web Interface**
- Open the web page in a browser.
- Upload an image to detect pipes.
- View live video detection and object counting.

### **2. Mobile Support**
- Navigate to `/mobile_live` for the mobile version.
- Upload images via your phone’s browser.

## Notes
- The `best.pt` model file is not included due to size limitations. Add it manually.
- The app runs on HTTPS. If you do not have SSL certificates (`cert.pem`, `key.pem`), use:
  ```sh
  flask run --host=0.0.0.0 --port=5000
  ```
  This will serve the app over HTTP.

## File Structure
```
PipeDetection/
│── app.py
│── requirements.txt
│── README.md
│── uploads/             # Folder for uploaded images
│── results/             # Folder for processed images
│── videos/              # Folder for recorded videos
│── templates/           # HTML templates
│   ├── index.html
│   ├── index_mobile.html
│   ├── display.html
│   ├── display_mobile.html
│   ├── live.html
│   ├── mobile_live.html
│── static/              # CSS and JS files
│── best.pt              # YOLO model (DO NOT UPLOAD TO GITHUB)
│── cert.pem             # SSL Certificate (DO NOT UPLOAD TO GITHUB)
│── key.pem              # SSL Key (DO NOT UPLOAD TO GITHUB)
│── .gitignore
```

## Contributing
Feel free to submit pull requests and report issues.

## License
This project is licensed under the MIT License. See `LICENSE` for details.