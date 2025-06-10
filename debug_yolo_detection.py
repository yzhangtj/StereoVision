#!/usr/bin/env python3
"""
Debug YOLO Detection Script

This script helps debug why YOLOv8 might be missing objects by:
1. Showing all YOLO classes
2. Running detection at multiple confidence levels
3. Displaying all raw detections before filtering
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def show_yolo_classes():
    """Display all YOLO classes that can be detected."""
    model = YOLO('yolov8n.pt')
    print("🏷️  YOLO Classes Available:")
    print("=" * 50)
    for i, class_name in model.names.items():
        print(f"{i:2d}: {class_name}")
    print("=" * 50)
    return model


def debug_detection(image_path, confidence_levels=[0.1, 0.2, 0.3, 0.5]):
    """Run detection at multiple confidence levels to see what's being filtered out."""
    model = YOLO('yolov8n.pt')
    
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = image_path
    
    print(f"\n🔍 Debug Detection Analysis")
    print(f"Image shape: {image.shape}")
    
    all_results = {}
    
    for conf in confidence_levels:
        print(f"\n📊 Confidence Level: {conf}")
        print("-" * 30)
        
        # Run detection
        results = model(image, conf=conf, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'area': (x2-x1) * (y2-y1)
                    })
        
        all_results[conf] = detections
        
        print(f"Found {len(detections)} objects:")
        for det in sorted(detections, key=lambda x: x['confidence'], reverse=True):
            bbox = det['bbox']
            print(f"  {det['class']:12} {det['confidence']:.3f} [{bbox[0]:3d},{bbox[1]:3d},{bbox[2]:3d},{bbox[3]:3d}] area:{det['area']:5d}")
    
    return all_results, model


def create_debug_visualization(image_path, confidence=0.2):
    """Create visualization showing all detections with labels."""
    model = YOLO('yolov8n.pt')
    
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = image_path
    
    # Run detection with low confidence
    results = model(image, conf=confidence, verbose=False)
    
    annotated_image = image.copy()
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # Color based on confidence
                if conf >= 0.5:
                    color = (0, 255, 0)  # Green - high confidence
                elif conf >= 0.3:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 128, 255)  # Orange - low confidence
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Create label
                label = f"{class_name} {conf:.2f}"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1-25), (x1+label_size[0]+5, y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_image, label, (x1+2, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_image


def main():
    # Show all available YOLO classes
    model = show_yolo_classes()
    
    # Test image path
    image_path = "test/000040/left.png"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n🖼️  Analyzing image: {image_path}")
    
    # Debug detection at multiple confidence levels
    results, model = debug_detection(image_path)
    
    # Create debug visualization
    debug_image = create_debug_visualization(image_path, confidence=0.1)
    
    # Save debug image
    output_path = "debug_yolo_detection.png"
    cv2.imwrite(output_path, debug_image)
    print(f"\n📸 Debug visualization saved: {output_path}")
    
    # Show the image
    cv2.imshow("Debug YOLO Detection (Green=High, Yellow=Med, Orange=Low conf)", debug_image)
    print(f"\n👁️  Press any key to close the window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Summary analysis
    print(f"\n📈 Summary Analysis:")
    print("=" * 50)
    for conf, detections in results.items():
        unique_classes = set(det['class'] for det in detections)
        print(f"Confidence {conf:3.1f}: {len(detections):2d} objects, {len(unique_classes):2d} unique classes")
        if detections:
            classes_str = ", ".join(sorted(unique_classes))
            print(f"             Classes: {classes_str}")


if __name__ == "__main__":
    main() 