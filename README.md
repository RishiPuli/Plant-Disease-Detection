# 🌱 Plant Disease Detection System

An intelligent AI-powered web application that can detect plant diseases from leaf images. This system uses deep learning to classify plant leaves into three categories: **Healthy**, **Powdery Mildew**, and **Rust Disease**.

## 🎯 What This Project Does

This application helps farmers, gardeners, and plant enthusiasts:
- **Quickly identify plant diseases** from photos of leaves
- **Get instant results** with confidence scores
- **Track prediction history** over time
- **View analytics** of disease patterns
- **Make informed decisions** about plant care

## 🚀 Features

- **📸 Image Upload**: Drag & drop or click to upload leaf images
- **🤖 AI Analysis**: Advanced deep learning model for accurate disease detection
- **📊 Real-time Results**: Instant classification with confidence scores
- **📈 Analytics Dashboard**: Visual charts showing prediction trends
- **💾 History Tracking**: Save and review all previous predictions
- **🌐 Web Interface**: User-friendly web application
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices

## 🛠️ System Requirements

Before you start, make sure you have:

- **Windows 10/11** (or macOS/Linux)
- **Python 3.8 or higher** (3.9+ recommended)
- **At least 4GB RAM** (8GB recommended)
- **2GB free disk space**
- **Internet connection** (for downloading dependencies)

## 📦 Installation Guide

### Step 1: Download the Project

1. **Download** this project folder to your computer
2. **Extract** the ZIP file if it's compressed
3. **Open Command Prompt** (Windows) or Terminal (Mac/Linux)

### Step 2: Navigate to the Project Folder

```bash
cd "path/to/Plant-Disease-Detection/plant_disease_detection"
```

**Example for Windows:**
```bash
cd "C:\Users\YourName\Downloads\Plant-Disease-Detection\plant_disease_detection"
```

### Step 3: Install Python Dependencies

**For Windows:**
```bash
pip install -r requirements.txt
pip install -r web_requirements.txt
```

**If you get errors, try:**
```bash
python -m pip install -r requirements.txt
python -m pip install -r web_requirements.txt
```

### Step 4: Verify Installation

Run this command to check if everything is installed correctly:
```bash
python -c "import tensorflow, flask, cv2; print('✅ All packages installed successfully!')"
```

## 🚀 How to Run the Application

### Method 1: Simple Start (Recommended)

1. **Open Command Prompt/Terminal** in the project folder
2. **Run the application:**
   ```bash
   python app.py
   ```
3. **Open your web browser** and go to: `http://localhost:5000`
4. **Start using the application!**

### Method 2: If Method 1 Doesn't Work

1. **Try with Python module:**
   ```bash
   python -m flask run
   ```

2. **Or specify the host:**
   ```bash
   python app.py --host=0.0.0.0 --port=5000
   ```

## 📖 How to Use the Application

### 1. **Upload an Image**
- Click the upload area or drag & drop a leaf image
- Supported formats: JPG, JPEG
- Make sure the image shows clear leaf details

### 2. **Get Results**
- Click "Analyze Plant" button
- Wait a few seconds for AI processing
- View the classification result and confidence score

### 3. **Understand Results**
- **Healthy**: Green, uniform leaves with no spots
- **Powdery Mildew**: White powdery spots or fuzzy coating
- **Rust Disease**: Orange-brown spots or powdery pustules

### 4. **View Analytics**
- Check the dashboard for prediction trends
- Review your prediction history
- See disease distribution charts

## 🔧 Troubleshooting

### Common Issues and Solutions

**❌ "Python not found" error:**
- Install Python from [python.org](https://python.org)
- Make sure to check "Add Python to PATH" during installation

**❌ "pip not found" error:**
- Try: `python -m pip install package_name`
- Or reinstall Python with PATH option

**❌ "Module not found" errors:**
- Run: `pip install --upgrade pip`
- Then reinstall requirements: `pip install -r requirements.txt`

**❌ "Port already in use" error:**
- Close other applications using port 5000
- Or change port: `python app.py --port=8080`

**❌ "Model file not found" error:**
- Make sure the `models/best_model.keras` file exists
- If missing, contact the project maintainer

**❌ "Permission denied" error:**
- Run Command Prompt as Administrator
- Or check folder permissions

### Getting Help

If you're still having issues:
1. **Check Python version**: `python --version`
2. **Check pip version**: `pip --version`
3. **Try virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
Plant-Disease-Detection/
├── plant_disease_detection/
│   ├── app.py                 # Main web application
│   ├── train.py              # Model training script
│   ├── predict.py            # Prediction script
│   ├── requirements.txt      # Python dependencies
│   ├── web_requirements.txt  # Web app dependencies
│   ├── models/
│   │   └── best_model.keras  # Trained AI model
│   ├── templates/
│   │   └── index.html        # Web interface
│   ├── static/
│   │   ├── css/              # Styling
│   │   ├── js/               # JavaScript
│   │   └── uploads/          # Uploaded images
│   └── utils/                # Helper functions
└── README.md                 # This file
```

## 🧠 Technical Details

### AI Model Information
- **Architecture**: EfficientNetB0 with custom layers
- **Training Data**: Plant leaf images (Healthy, Powdery Mildew, Rust)
- **Accuracy**: High accuracy on test dataset
- **Input Size**: 512x512 pixels
- **Output**: 3-class classification with confidence scores

### Technologies Used
- **Backend**: Python Flask
- **AI Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV
- **Frontend**: HTML, CSS, JavaScript
- **Charts**: Plotly.js

## 📊 Performance Tips

### For Better Results:
1. **Use clear, well-lit photos**
2. **Focus on individual leaves**
3. **Avoid shadows and reflections**
4. **Ensure leaves are in focus**
5. **Use high-resolution images**

### For Faster Processing:
1. **Close unnecessary applications**
2. **Use smaller image files** (under 5MB)
3. **Ensure good internet connection**
4. **Restart the application** if it becomes slow

## 🤝 Contributing

Want to improve this project?
1. **Report bugs** with detailed descriptions
2. **Suggest new features**
3. **Share plant images** for model improvement
4. **Improve documentation**

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Dataset providers** for plant disease images
- **Open source community** for libraries and tools
- **Researchers** in plant pathology and computer vision

## 📞 Support

If you need help:
1. **Check this README** for common solutions
2. **Review error messages** carefully
3. **Try the troubleshooting section**
4. **Contact your instructor** or project maintainer

---

**Happy Plant Disease Detection! 🌿🔬**

*Made with ❤️ for plant lovers and AI enthusiasts*
