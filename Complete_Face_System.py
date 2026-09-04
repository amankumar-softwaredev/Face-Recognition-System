"""
COMPLETE FACE RECOGNITION SYSTEM
Developed by: AMAN KUMAR
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path   
import numpy as np   
from PIL import Image, ImageDraw, ImageFont
from scrfd import SCRFD, Threshold  
from arcface import ArcFace

BASE_DIR = Path(__file__).resolve().parent

scrfd_models_dir = BASE_DIR / "models"
arcface_models_dir = BASE_DIR / "arcface_models"
single_person_dir = BASE_DIR / "Single_Person"
group_images_dir = BASE_DIR / "Group_image"
detection_images_dir = BASE_DIR / "Detection_Image"
output_detection_dir = BASE_DIR / "Image_FaceDetection"
embeddings_dir = BASE_DIR / "embeddings"

single_person_dir.mkdir(exist_ok=True)
group_images_dir.mkdir(exist_ok=True)
detection_images_dir.mkdir(exist_ok=True)
output_detection_dir.mkdir(exist_ok=True)
embeddings_dir.mkdir(exist_ok=True)

def load_models():
    scrfd_model_files = sorted(scrfd_models_dir.glob("*.onnx"))
    arcface_model_files = sorted(arcface_models_dir.glob("*.tflite"))
    
    if not scrfd_model_files or not arcface_model_files:
        raise FileNotFoundError("Models not found!")
    
    print("Loading models...")
    face_detector = SCRFD.from_path(str(scrfd_model_files[0]))
    face_rec = ArcFace.ArcFace(model_path=str(arcface_model_files[0]))
    threshold = Threshold(probability=0.65)
    
    print("✓ Models loaded!\n")
    return face_detector, face_rec, threshold

def part1_single_person_training(face_detector, face_rec, threshold):
    print("\n" + "="*70)
    print("PART 1: SINGLE PERSON - MULTIPLE PHOTOS → AVERAGE EMBEDDING")
    print("="*70)
    
    image_paths = []
    image_paths.extend(sorted(single_person_dir.glob("*.jpeg")))
    image_paths.extend(sorted(single_person_dir.glob("*.jpg")))
    image_paths.extend(sorted(single_person_dir.glob("*.png")))
    
    if len(image_paths) == 0:
        print(f"\n❌ No images found in {single_person_dir}")
        print("Please add person's photos to Single_Person/ folder!")
        return
    
    print(f"\nFound {len(image_paths)} image(s):")
    for idx, img in enumerate(image_paths, 1):
        print(f"  {idx}. {img.name}")
    
    person_name = input("\nEnter person name: ").strip()
    if not person_name:
        print("❌ Name cannot be empty!")
        return
    
    embeddings_list = []
    processed_images = []
    
    print(f"\n{'='*70}")
    print(f"PROCESSING PHOTOS FOR: {person_name.upper()}")
    print(f"{'='*70}")
    
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] {image_path.name}")
        
        try:
            image = Image.open(image_path).convert("RGB")
            faces = face_detector.detect(image, threshold=threshold)
            
            if len(faces) == 0:
                print("  ❌ No face detected, skipping...")
                continue
            
            best_face = max(faces, key=lambda f: float(f.probability))
            bbox = best_face.bbox
            confidence = float(best_face.probability)
            
            x1 = max(0, int(round(bbox.upper_left.x)))
            y1 = max(0, int(round(bbox.upper_left.y)))
            x2 = min(image.width, int(round(bbox.lower_right.x)))
            y2 = min(image.height, int(round(bbox.lower_right.y)))
            
            if x2 <= x1 or y2 <= y1:
                print("  ❌ Invalid bbox, skipping...")
                continue
            
            face_crop = image.crop((x1, y1, x2, y2))
            crop_path = embeddings_dir / f"{person_name}_{idx}_crop.jpg"
            face_crop.save(crop_path, quality=95)
            
            embedding = face_rec.calc_emb(str(crop_path))
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            
            if embedding.size != 512:
                print(f"  ❌ Invalid embedding size: {embedding.size}")
                continue
            
            embedding = embedding / np.linalg.norm(embedding)
            
            embeddings_list.append(embedding)
            processed_images.append(image_path.name)
            
            print(f"  ✓ Face detected (confidence: {confidence:.3f})")
            print(f"  ✓ Embedding generated")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    if len(embeddings_list) == 0:
        print("\n❌ No embeddings generated!")
        return
    
    print(f"\n{'='*70}")
    print("CALCULATING AVERAGE EMBEDDING")
    print(f"{'='*70}")
    print(f"Total images processed: {len(embeddings_list)}")
    print(f"Successfully used images:")
    for img_name in processed_images:
        print(f"  ✓ {img_name}")
    
    embeddings_matrix = np.vstack(embeddings_list)
    average_vector = np.mean(embeddings_matrix, axis=0)
    
    average_vector = average_vector / np.linalg.norm(average_vector)
    average_vector = average_vector.astype(np.float32)
    
    print(f"\n✓ Average embedding calculated")
    print(f"  Shape: {average_vector.shape}")
    print(f"  Norm: {np.linalg.norm(average_vector):.6f}")
    
    embedding_npy = embeddings_dir / f"{person_name}_embedding.npy"
    embedding_txt = embeddings_dir / f"{person_name}_embedding.txt"
    
    np.save(embedding_npy, average_vector)
    np.savetxt(embedding_txt, average_vector, fmt="%.8f")
    
    print(f"\n✅ Embedding saved:")
    print(f"  - {embedding_npy.name}")
    print(f"  - {embedding_txt.name}")
    
    return person_name

def part2_group_face_selection(face_detector, face_rec, threshold):
    print("\n" + "="*70)
    print("PART 2: GROUP PHOTOS - FACE SELECTION & AVERAGE EMBEDDING")
    print("="*70)
    
    group_images = []
    group_images.extend(sorted(group_images_dir.glob("*.jpeg")))
    group_images.extend(sorted(group_images_dir.glob("*.jpg")))
    group_images.extend(sorted(group_images_dir.glob("*.png")))
    
    if len(group_images) == 0:
        print(f"\n❌ No images in {group_images_dir}")
        print("Please add group photos first!")
        return
    
    print(f"\nFound {len(group_images)} group photo(s)")
    
    all_photos_data = []
    
    for photo_idx, image_path in enumerate(group_images, 1):
        print(f"\n{'='*70}")
        print(f"PROCESSING GROUP PHOTO {photo_idx}: {image_path.name}")
        print(f"{'='*70}")
        
        image = Image.open(image_path).convert("RGB")
        faces = face_detector.detect(image, threshold=threshold)
        
        if len(faces) == 0:
            print("❌ No faces detected in this image!")
            continue
        
        print(f"✓ Detected {len(faces)} face(s)")
        
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        photo_faces = {
            'photo_idx': photo_idx,
            'photo_name': image_path.name,
            'faces': {}
        }
        
        for face_num, face in enumerate(faces, 1):
            bbox = face.bbox
            confidence = float(face.probability)
            
            x1 = max(0, int(round(bbox.upper_left.x)))
            y1 = max(0, int(round(bbox.upper_left.y)))
            x2 = min(image.width, int(round(bbox.lower_right.x)))
            y2 = min(image.height, int(round(bbox.lower_right.y)))
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=6)
            
            label = f"Face {face_num}"
            text_y = y1 - 50 if y1 > 50 else y1 + 10
            draw.text((x1, text_y), label, fill="red", font=font)
            
            print(f"  Face {face_num}: Confidence={confidence:.3f}, BBox=[{x1},{y1},{x2},{y2}]")
            
            face_crop = image.crop((x1, y1, x2, y2))
            crop_path = embeddings_dir / f"group{photo_idx}_face{face_num}_crop.jpg"
            face_crop.save(crop_path, quality=95)
            
            embedding = face_rec.calc_emb(str(crop_path))
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            
            if embedding.size != 512:
                print(f"  ❌ Invalid embedding size for Face {face_num}")
                continue
            
            embedding = embedding / np.linalg.norm(embedding)
            
            photo_faces['faces'][face_num] = {
                'embedding': embedding,
                'crop_path': crop_path,
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence
            }
        
        output_path = group_images_dir / f"annotated_photo{photo_idx}_{image_path.name}"
        image.save(output_path)
        print(f"\n✓ Annotated image saved: {output_path.name}")
        
        all_photos_data.append(photo_faces)
    
    if len(all_photos_data) == 0:
        print("\n❌ No faces detected in any photo!")
        return
    
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL DETECTED FACES")
    print(f"{'='*70}")
    
    for photo_data in all_photos_data:
        photo_idx = photo_data['photo_idx']
        photo_name = photo_data['photo_name']
        num_faces = len(photo_data['faces'])
        print(f"\nPhoto {photo_idx} ({photo_name}): {num_faces} face(s)")
        for face_num in photo_data['faces'].keys():
            print(f"  - Face {face_num}")
    
    print(f"\n{'='*70}")
    print("SELECT FACES FOR SAME PERSON")
    print(f"{'='*70}")
    print("Example: Photo 1 ka Face 2, Photo 2 ka Face 5 same person hai")
    print("Format: 1-2, 2-5  (Photo-Face pairs, comma separated)")
    
    person_name = input("\nEnter person name: ").strip()
    if not person_name:
        person_name = "unknown_person"
    
    print(f"\nSelect faces for {person_name.upper()}")
    print("Enter face selections (e.g., 1-2,2-5,3-1):")
    
    selection = input("> ").strip()
    
    if not selection:
        print("❌ No selection made!")
        return
    
    selected_embeddings = []
    
    try:
        pairs = selection.split(',')
        for pair in pairs:
            pair = pair.strip()
            photo_num, face_num = map(int, pair.split('-'))
            
            photo_data = None
            for pd in all_photos_data:
                if pd['photo_idx'] == photo_num:
                    photo_data = pd
                    break
            
            if photo_data is None:
                print(f"❌ Photo {photo_num} not found!")
                continue
            
            if face_num not in photo_data['faces']:
                print(f"❌ Face {face_num} not found in Photo {photo_num}!")
                continue
            
            embedding = photo_data['faces'][face_num]['embedding']
            selected_embeddings.append(embedding)
            print(f"✓ Added: Photo {photo_num} - Face {face_num}")
    
    except Exception as e:
        print(f"❌ Error parsing selection: {str(e)}")
        return
    
    if len(selected_embeddings) == 0:
        print("\n❌ No valid faces selected!")
        return
    
    print(f"\n{'='*70}")
    print("CALCULATING AVERAGE EMBEDDING")
    print(f"{'='*70}")
    print(f"Total selected faces: {len(selected_embeddings)}")
    
    embeddings_matrix = np.vstack(selected_embeddings)
    average_vector = np.mean(embeddings_matrix, axis=0)
    
    average_vector = average_vector / np.linalg.norm(average_vector)
    average_vector = average_vector.astype(np.float32)
    
    print(f"✓ Average embedding calculated")
    print(f"  Shape: {average_vector.shape}")
    print(f"  Norm: {np.linalg.norm(average_vector):.6f}")
    
    embedding_npy = embeddings_dir / f"{person_name}_embedding.npy"
    embedding_txt = embeddings_dir / f"{person_name}_embedding.txt"
    
    np.save(embedding_npy, average_vector)
    np.savetxt(embedding_txt, average_vector, fmt="%.8f")
    
    print(f"\n✅ Embedding saved:")
    print(f"  - {embedding_npy.name}")
    print(f"  - {embedding_txt.name}")
    
    return person_name

def part3_detection_on_new_images(face_detector, face_rec, threshold):
    print("\n" + "="*70)
    print("PART 3: DETECTION ON NEW IMAGES")
    print("="*70)
    
    embedding_files = list(embeddings_dir.glob("*_embedding.npy"))
    
    if len(embedding_files) == 0:
        print("\n❌ No embeddings found!")
        print("Please create embeddings first (Part 1 or Part 2)")
        return
    
    print(f"\nAvailable embeddings:")
    for idx, emb_file in enumerate(embedding_files, 1):
        person_name = emb_file.stem.replace('_embedding', '')
        print(f"  {idx}. {person_name}")
    
    try:
        choice = int(input(f"\nSelect embedding to use (1-{len(embedding_files)}): "))
        if choice < 1 or choice > len(embedding_files):
            print("❌ Invalid choice!")
            return
        
        selected_embedding_file = embedding_files[choice - 1]
        person_name = selected_embedding_file.stem.replace('_embedding', '')
        
    except ValueError:
        print("❌ Invalid input!")
        return
    
    reference_embedding = np.load(selected_embedding_file).astype(np.float32).reshape(-1)
    reference_embedding = reference_embedding / np.linalg.norm(reference_embedding)
    
    print(f"\n✓ Loaded embedding for: {person_name.upper()}")
    
    detection_images = []
    detection_images.extend(sorted(detection_images_dir.glob("*.jpeg")))
    detection_images.extend(sorted(detection_images_dir.glob("*.jpg")))
    detection_images.extend(sorted(detection_images_dir.glob("*.png")))
    
    if len(detection_images) == 0:
        print(f"\n❌ No images in {detection_images_dir}")
        print("Please add images for detection!")
        return
    
    print(f"\nFound {len(detection_images)} image(s) in Detection_Image folder:")
    for idx, img in enumerate(detection_images, 1):
        print(f"  {idx}. {img.name}")
    
    print(f"\n{'='*70}")
    print("SELECT IMAGES TO PROCESS")
    print(f"{'='*70}")
    print("Options:")
    print("  1. Process ALL images")
    print("  2. Select by image NUMBER (e.g., 1,3,5)")
    print("  3. Select by image NAME (e.g., photo1.jpg)")
    
    process_choice = input("\nEnter choice (1/2/3): ").strip()
    
    images_to_process = []
    
    if process_choice == "1":
        images_to_process = detection_images
        print("\n✓ Will process ALL images")
    
    elif process_choice == "2":
        print("\nEnter image numbers (comma separated)")
        print("Example: 1,3,5")
        selection = input("> ").strip()
        
        try:
            selected_indices = [int(x.strip()) for x in selection.split(',')]
            for idx in selected_indices:
                if 1 <= idx <= len(detection_images):
                    images_to_process.append(detection_images[idx - 1])
                else:
                    print(f"⚠️ Image {idx} out of range, skipping...")
            
            if len(images_to_process) == 0:
                print("❌ No valid images selected!")
                return
            
            print(f"\n✓ Will process {len(images_to_process)} selected image(s)")
        
        except ValueError:
            print("❌ Invalid input format!")
            return
    
    elif process_choice == "3":
        print("\nEnter image name(s) (comma separated)")
        print("Example: photo1.jpg,party.jpg")
        selection = input("> ").strip()
        
        image_names = [x.strip() for x in selection.split(',')]
        image_dict = {img.name: img for img in detection_images}
        
        for img_name in image_names:
            if img_name in image_dict:
                images_to_process.append(image_dict[img_name])
                print(f"✓ Added: {img_name}")
            else:
                print(f"⚠️ Image '{img_name}' not found, skipping...")
        
        if len(images_to_process) == 0:
            print("❌ No valid images selected!")
            return
        
        print(f"\n✓ Will process {len(images_to_process)} selected image(s)")
    
    else:
        print("❌ Invalid choice!")
        return
    
    MATCH_THRESHOLD = 0.65
    
    print(f"\n{'='*70}")
    print("STARTING DETECTION")
    print(f"{'='*70}")
    
    for img_idx, image_path in enumerate(images_to_process, 1):
        print(f"\n[{img_idx}/{len(images_to_process)}] {image_path.name}")
        print("-"*70)
        
        try:
            image = Image.open(image_path).convert("RGB")
            faces = face_detector.detect(image, threshold=threshold)
            
            if len(faces) == 0:
                print("❌ No faces detected")
                continue
            
            print(f"✓ Detected {len(faces)} face(s)")
            
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                font = ImageFont.load_default()
            
            match_found = False
            
            for face_num, face in enumerate(faces, 1):
                bbox = face.bbox
                confidence = float(face.probability)
                
                x1 = max(0, int(round(bbox.upper_left.x)))
                y1 = max(0, int(round(bbox.upper_left.y)))
                x2 = min(image.width, int(round(bbox.lower_right.x)))
                y2 = min(image.height, int(round(bbox.lower_right.y)))
                
                face_crop = image.crop((x1, y1, x2, y2))
                crop_path = embeddings_dir / f"detection_temp_{img_idx}_{face_num}.jpg"
                face_crop.save(crop_path, quality=95)
                
                embedding = face_rec.calc_emb(str(crop_path))
                embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                embedding = embedding / np.linalg.norm(embedding)
                
                similarity = float(np.dot(reference_embedding, embedding))
                
                if similarity >= MATCH_THRESHOLD:
                    color = "green"
                    label = f"{person_name.upper()}"
                    line_width = 8
                    match_found = True
                    print(f"  Face {face_num}: ✅ MATCH! Similarity={similarity:.4f}")
                else:
                    color = "yellow"
                    label = f"Face {face_num}"
                    line_width = 4
                    print(f"  Face {face_num}: ❌ No match. Similarity={similarity:.4f}")
                
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                
                text_y = y1 - 50 if y1 > 50 else y1 + 10
                draw.text((x1, text_y), label, fill=color, font=font)
                
                crop_path.unlink()
            
            if match_found:
                output_filename = f"FOUND_{person_name}_{image_path.name}"
            else:
                output_filename = f"NOT_FOUND_{image_path.name}"
            
            output_path = output_detection_dir / output_filename
            image.save(output_path)
            
            if match_found:
                print(f"\n✅ {person_name.upper()} FOUND in this image!")
            else:
                print(f"\n❌ {person_name.upper()} NOT FOUND in this image")
            
            print(f"✓ Result saved: {output_filename}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue
    
    print(f"\n{'='*70}")
    print("✅ DETECTION COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved in: {output_detection_dir.name}/")

def main():
    print("\n" + "="*70)
    print("🎯 COMPLETE FACE RECOGNITION SYSTEM")
    print("="*70)
    
    face_detector, face_rec, threshold = load_models()
    
    while True:
        print("\n" + "-"*70)
        print("SELECT OPTION:")
        print("-"*70)
        print("1. Single Person (Multiple Photos) → Average Embedding")
        print("2. Group Photos (Manual Face Selection) → Average Embedding")
        print("3. Detection on New Images (using saved embedding)")
        print("4. Exit")
        print("-"*70)
        
        choice = input("\nEnter choice (1/2/3/4): ").strip()
        
        if choice == "1":
            part1_single_person_training(face_detector, face_rec, threshold)
        
        elif choice == "2":
            part2_group_face_selection(face_detector, face_rec, threshold)
        
        elif choice == "3":
            part3_detection_on_new_images(face_detector, face_rec, threshold)
        
        elif choice == "4":
            print("\n👋 Exiting... Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice!")

if __name__ == "__main__":
    main()
