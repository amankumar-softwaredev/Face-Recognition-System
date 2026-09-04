# 🛠️ Setup Guide

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Face-Recognition-System.git
cd Face-Recognition-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pillow tensorflow onnxruntime
```

### 3. Download Models

#### SCRFD Model (Face Detection)
1. Download from: [SCRFD Model Link]
2. Place `.onnx` file in `models/` folder

#### ArcFace Model (Face Embedding)
1. Download from: [ArcFace Model Link]
2. Place `.tflite` file in `arcface_models/` folder

### 4. Create Folders

The script will auto-create folders, but you can create them manually:

```bash
mkdir Single_Person
mkdir Group_image
mkdir Detection_Image
mkdir Image_FaceDetection
mkdir embeddings
mkdir models
mkdir arcface_models
```

### 5. Run the System

```bash
python Complete_Face_System.py
```

---

## 📦 Folder Structure After Setup

```
Face-Recognition-System/
├── Complete_Face_System.py
├── README.md
├── requirements.txt
├── .gitignore
├── Single_Person/          (empty - add your photos)
├── Group_image/            (empty - add your photos)
├── Detection_Image/        (empty - add your photos)
├── Image_FaceDetection/    (empty - outputs here)
├── embeddings/             (empty - auto-generated)
├── models/                 (add SCRFD .onnx file)
└── arcface_models/         (add ArcFace .tflite file)
```

---

## ✅ Verification

Test if setup is correct:

```bash
python Complete_Face_System.py
```

You should see:
```
Loading models...
✓ Models loaded!

SELECT OPTION:
1. Single Person (Multiple Photos) → Average Embedding
2. Group Photos (Manual Face Selection) → Average Embedding
3. Detection on New Images (using saved embedding)
4. Exit
```

If you see errors:
- Check if models are in correct folders
- Check if dependencies are installed
- See Troubleshooting section in README.md

---

## 🎯 Quick Test

### Test with Sample Images

1. Add 3-4 photos of yourself to `Single_Person/`
2. Run: `python Complete_Face_System.py`
3. Select Option 1
4. Enter your name
5. Check `embeddings/` folder for your `.npy` file

---

## 🔧 Model Download Links

**Note:** Models are not included in repository due to size.

### SCRFD Model
- Provider: InsightFace
- Format: `.onnx`
- Size: ~2.5 MB
- Link: [Add actual link]

### ArcFace Model
- Provider: InsightFace
- Format: `.tflite`
- Size: ~6 MB
- Link: [Add actual link]

---

## 💻 System Requirements

- **Python**: 3.7 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space
- **OS**: Windows, Linux, macOS

---

## 🐛 Common Issues

### Issue: "Models not found!"
**Solution:** Ensure `.onnx` and `.tflite` files are in correct folders

### Issue: TensorFlow warnings
**Solution:** Ignore warnings, they don't affect functionality

### Issue: Slow processing
**Solution:** Use CPU mode or reduce image resolution

---

**Setup complete! Ready to use!** 🎉
