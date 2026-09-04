# Face Recognition System

A complete face recognition system with training and detection capabilities using SCRFD and ArcFace models.

Developer: AMAN KUMAR

---

## Features

- Single Person Training - Train with multiple photos of one person
- Group Photo Training - Manual face selection from group photos
- Face Detection - Detect and recognize faces in new images
- High Accuracy - ArcFace embeddings with 512-dimensional vectors
- Flexible Selection - Process all images or select specific ones
- Visual Output - Green boxes for matches, yellow for non-matches

## Quick Start

### 1. Installation

```bash
pip install numpy pillow scrfd arcface tensorflow
```

### 2. Setup

Download the required models:
- SCRFD model → Place in models/ folder
- ArcFace model → Place in arcface_models/ folder

Note: Model files are not included in this repository due to size constraints. Please download them separately.

### 3. Run

```bash
python Complete_Face_System.py
```

## Folder Structure

```
Face-Recognition-System/
├── Complete_Face_System.py    # Main script
├── Single_Person/             # Add single person training photos here
├── Group_image/               # Add group photos for training here
├── Detection_Image/           # Add images to detect faces (input)
├── Image_FaceDetection/       # Detection results will be saved here (output)
├── embeddings/                # Generated embeddings
├── models/                    # SCRFD face detection model (download separately)
└── arcface_models/            # ArcFace embedding model (download separately)
```

Note: Detection_Image and Image_FaceDetection folders are kept empty intentionally:
- Detection_Image: Add your test images here
- Image_FaceDetection: Detection results will automatically be saved here

## Usage

### Option 1: Single Person Training

Train with multiple photos of one person to create an average embedding.

Steps:
1. Add 3-7 photos to Single_Person/ folder
2. Run script and select Option 1
3. Enter person name
4. Done! Embedding created

Example:
```
Single_Person/
├── john_1.jpg
├── john_2.jpg
└── john_3.jpg

Output: john_embedding.npy
```

### Option 2: Group Photo Training

Select specific faces from group photos to create an average embedding.

Steps:
1. Add 2-3 group photos to Group_image/ folder
2. Run script and select Option 2
3. Check annotated images (faces numbered)
4. Enter person name
5. Select faces: 1-3,2-5,3-2 (Photo-Face pairs)
6. Done! Embedding created

Example:
```
Group_image/
├── party1.jpg (8 people)
├── party2.jpg (6 people)
└── party3.jpg (10 people)

Selection: 1-3,2-5,3-7
Means: Photo 1 Face 3 + Photo 2 Face 5 + Photo 3 Face 7
Output: person_embedding.npy
```

### Option 3: Face Detection

Detect faces in new images using saved embeddings.

Steps:
1. Add test images to Detection_Image/ folder
2. Run script and select Option 3
3. Select saved embedding
4. Choose:
   - Process ALL images
   - Select by NUMBER (e.g., 1,3,5)
   - Select by NAME (e.g., photo1.jpg,party.jpg)
5. Done! Results in Image_FaceDetection/ folder

Output:
- Green box = Person found (match)
- Yellow box = No match

Example:
```
Detection_Image/
├── test1.jpg
├── test2.jpg
└── test3.jpg

Results:
├── FOUND_john_test1.jpg     (Green outline)
├── NOT_FOUND_test2.jpg      (Yellow outline)
└── FOUND_john_test3.jpg     (Green outline)
```

## Technical Details

### Models
- SCRFD: Face detection model
- ArcFace: Face embedding model (512-dimensional vectors)

### Thresholds
- Face Detection: 0.65 (65% confidence)
- Face Matching: 0.65 (cosine similarity threshold)

### Similarity Levels
- 0.65 - 0.70: Good match
- 0.70 - 0.80: Strong match
- 0.80+: Very strong match

## Workflow

```
Training:
Photos → Face Detection → Crop → Embedding → Average → Save

Detection:
Test Image → Face Detection → Embedding → Compare → Match/No Match → Output
```

## Example Use Cases

### 1. Attendance System
- Train with employee photos
- Detect in daily camera captures
- Automatic attendance marking

### 2. Photo Organization
- Train with person's photos
- Find all photos of that person in albums
- Auto-tag and organize

### 3. Security System
- Train with authorized persons
- Detect in camera feeds
- Alert on unauthorized access

### 4. Event Photography
- Train from group photos
- Find specific person in all event photos
- Quick photo delivery

## Tips for Best Results

### Training
- Use 3-7 photos per person
- Include different angles (front, side, slight tilt)
- Good lighting and clear faces
- Variety in expressions (smiling, serious)

### Detection
- Use high-quality images
- Ensure good lighting
- Clear, unobstructed faces

## Configuration

You can adjust thresholds in the code:

```python
# Face detection confidence
threshold = Threshold(probability=0.65)

# Face matching threshold
MATCH_THRESHOLD = 0.65
```

## Troubleshooting

### "No faces detected"
- Check image quality
- Ensure proper lighting
- Face should be clearly visible

### "Person not found" (but visible)
- Lower threshold to 0.55-0.60
- Add more training photos
- Try different angles in training

### "Wrong person detected"
- Increase threshold to 0.70
- Add more training photos
- Use better quality images

## Dependencies

```
numpy
pillow (PIL)
scrfd
arcface
tensorflow
```

## License

MIT License - Free to use and modify

## Acknowledgments

- SCRFD for face detection
- ArcFace for face embeddings
- TensorFlow Lite for model inference

## Contact

Developer: AMAN KUMAR

For issues, suggestions, or contributions, please open an issue on GitHub.

---

Made with ❤️ by AMAN KUMAR
