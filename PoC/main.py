import cv2 as cv
import numpy as np
import os
from ultralytics import YOLO
from collections import defaultdict
import math


class EggTracker:
    """
    Optimized egg tracking using IoU (Intersection over Union) matching
    """
    def __init__(self, max_disappeared=5):  # Reduced from 10 to 5
        self.next_id = 0
        self.tracks = {}  # track_id: {bbox, last_seen, size_history}
        self.max_disappeared = max_disappeared
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, x2, y2 = box1
        x1_prime, y1_prime, x2_prime, y2_prime = box2
        
        # Calculate intersection
        xi1 = max(x1, x1_prime)
        yi1 = max(y1, y1_prime)
        xi2 = min(x2, x2_prime)
        yi2 = min(y2, y2_prime)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_prime - x1_prime) * (y2_prime - y1_prime)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of (x1, y1, x2, y2, confidence) tuples
        """
        if len(detections) == 0:
            # Mark all tracks as disappeared
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['last_seen'] += 1
                if self.tracks[track_id]['last_seen'] > self.max_disappeared:
                    del self.tracks[track_id]
            return {}
        
        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            for det in detections:
                x1, y1, x2, y2, conf = det
                self.tracks[self.next_id] = {
                    'bbox': (x1, y1, x2, y2),
                    'last_seen': 0,
                    'size_history': []
                }
                self.next_id += 1
            return {i: det[:4] for i, det in enumerate(detections)}
        
        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.calculate_iou(
                    self.tracks[track_id]['bbox'], det[:4]
                )
        
        # Simple greedy assignment with optimized matching
        matched_indices = []
        matched_track_ids = []
        iou_threshold = 0.2  # Lowered from 0.3 for faster matching
        
        # Find best matches - early exit optimization
        max_matches = min(len(track_ids), len(detections))
        for _ in range(max_matches):
            max_iou = np.max(iou_matrix)
            if max_iou < iou_threshold:
                break
            
            max_row, max_col = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matched_track_ids.append(track_ids[max_row])
            matched_indices.append(max_col)
            
            # Zero out this row and column
            iou_matrix[max_row, :] = 0
            iou_matrix[:, max_col] = 0
        
        # Update matched tracks
        active_tracks = {}
        for track_id, det_idx in zip(matched_track_ids, matched_indices):
            x1, y1, x2, y2, conf = detections[det_idx]
            self.tracks[track_id]['bbox'] = (x1, y1, x2, y2)
            self.tracks[track_id]['last_seen'] = 0
            active_tracks[track_id] = (x1, y1, x2, y2)
        
        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_indices:
                x1, y1, x2, y2, conf = det
                self.tracks[self.next_id] = {
                    'bbox': (x1, y1, x2, y2),
                    'last_seen': 0,
                    'size_history': []
                }
                active_tracks[self.next_id] = (x1, y1, x2, y2)
                self.next_id += 1
        
        # Update disappeared counter for unmatched tracks
        for track_id in track_ids:
            if track_id not in matched_track_ids:
                self.tracks[track_id]['last_seen'] += 1
                if self.tracks[track_id]['last_seen'] > self.max_disappeared:
                    del self.tracks[track_id]
        
        return active_tracks


def calculate_pixel_to_cm_ratio(conveyor_length_cm, conveyor_length_pixels):
    """
    Calculate the pixel to cm ratio based on conveyor belt dimensions
    """
    return conveyor_length_cm / conveyor_length_pixels


def calculate_egg_size(bbox, pixel_to_cm_ratio):
    """
    Calculate real-world size of egg from bounding box
    """
    x1, y1, x2, y2 = bbox
    width_pixels = x2 - x1
    height_pixels = y2 - y1
    
    width_cm = width_pixels * pixel_to_cm_ratio
    height_cm = height_pixels * pixel_to_cm_ratio
    
    return width_cm, height_cm


def detect_conveyor_belt_region(frame):
    """
    Detect the brown conveyor belt region in the image
    Returns the bounding box of the conveyor belt area
    """
    # Convert to HSV for better color detection
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Define brown color range for conveyor belt
    # Brown color ranges in HSV
    lower_brown1 = np.array([8, 50, 20])    # Lighter brown
    upper_brown1 = np.array([20, 255, 200])
    
    lower_brown2 = np.array([0, 50, 20])    # Darker brown/reddish
    upper_brown2 = np.array([10, 255, 150])
    
    # Create masks for brown colors
    mask1 = cv.inRange(hsv, lower_brown1, upper_brown1)
    mask2 = cv.inRange(hsv, lower_brown2, upper_brown2)
    brown_mask = cv.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    brown_mask = cv.morphologyEx(brown_mask, cv.MORPH_CLOSE, kernel)
    brown_mask = cv.morphologyEx(brown_mask, cv.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv.findContours(brown_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: assume conveyor covers central 80% of frame
        height, width = frame.shape[:2]
        return (int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9))
    
    # Find the largest contour (likely the conveyor belt)
    largest_contour = max(contours, key=cv.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv.boundingRect(largest_contour)
    
    return (x, y, x + w, y + h)


def estimate_conveyor_length_pixels(frame_width):
    """
    Estimate conveyor length in pixels based on frame width
    Assuming conveyor takes up about 80% of frame width
    """
    return int(frame_width * 0.8)


def calculate_perspective_corrected_ratio_with_belt_detection(camera_height_cm, conveyor_length_cm, frame, conveyor_bbox=None):
    """
    Calculate pixel-to-cm ratio accounting for camera perspective and actual conveyor belt detection
    
    For a fixed camera setup:
    - Camera height: 70cm
    - Conveyor length: 75cm
    - Camera is looking down at an angle
    """
    frame_height, frame_width = frame.shape[:2]
    
    if conveyor_bbox is None:
        # Detect conveyor belt region
        conveyor_bbox = detect_conveyor_belt_region(frame)
    
    x1, y1, x2, y2 = conveyor_bbox
    conveyor_width_pixels = x2 - x1
    conveyor_height_pixels = y2 - y1
    
    # Calculate horizontal ratio based on actual detected conveyor width
    horizontal_ratio = conveyor_length_cm / conveyor_width_pixels
    
    # For vertical measurements, account for perspective foreshortening
    # Objects further from camera appear smaller due to perspective
    perspective_factor = 1.1  # Slight adjustment for perspective
    vertical_ratio = horizontal_ratio * perspective_factor
    
    return horizontal_ratio, vertical_ratio, conveyor_bbox


def calculate_perspective_corrected_ratio(camera_height_cm, conveyor_length_cm, frame_width, frame_height):
    """
    Calculate pixel-to-cm ratio accounting for camera perspective and field of view
    
    For a fixed camera setup:
    - Camera height: 70cm
    - Conveyor length: 75cm
    - Camera is looking down at an angle
    """
    # Estimate the actual field of view dimensions at the conveyor level
    # Using simple trigonometry and assuming camera covers the conveyor width
    
    # For top-down view, the conveyor takes up most of the frame width
    conveyor_width_pixels = estimate_conveyor_length_pixels(frame_width)
    
    # Calculate horizontal ratio (width)
    horizontal_ratio = conveyor_length_cm / conveyor_width_pixels
    
    # For vertical measurements, account for perspective foreshortening
    # Objects further from camera appear smaller due to perspective
    # Assuming camera angle creates some foreshortening effect
    perspective_factor = 1.1  # Slight adjustment for perspective
    vertical_ratio = horizontal_ratio * perspective_factor
    
    return horizontal_ratio, vertical_ratio


def calculate_egg_size_with_perspective(bbox, horizontal_ratio, vertical_ratio):
    """
    Calculate real-world size of egg from bounding box with perspective correction
    """
    x1, y1, x2, y2 = bbox
    width_pixels = x2 - x1
    height_pixels = y2 - y1
    
    width_cm = width_pixels * horizontal_ratio
    height_cm = height_pixels * vertical_ratio
    
    return width_cm, height_cm


# function to read model file that i trained
def read_model_file(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        # Load YOLO model
        model = YOLO(model_path)
        print(f"YOLO model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
#applying model on frames of video - optimized version
def apply_model_on_frame_with_tracking(model, frame, tracker, horizontal_ratio, vertical_ratio, conveyor_bbox=None):
    """
    Optimized: Apply YOLO model to detect eggs, track them, and calculate their sizes
    Only consider detections within the conveyor belt region
    """
    # Run YOLO inference with faster settings
    results = model(frame, verbose=False)  # Disable verbose output
    
    # Extract detections with early filtering
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                confidence = box.conf[0].cpu().numpy()
                # Skip low confidence detections early
                if confidence < 0.5:  # Confidence threshold
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Quick conveyor belt region check
                if conveyor_bbox is not None:
                    belt_x1, belt_y1, belt_x2, belt_y2 = conveyor_bbox
                    center_x = (x1 + x2) * 0.5  # Faster multiplication
                    center_y = (y1 + y2) * 0.5
                    
                    # Only include detections within conveyor belt region
                    if (belt_x1 <= center_x <= belt_x2 and belt_y1 <= center_y <= belt_y2):
                        detections.append((x1, y1, x2, y2, confidence))
                else:
                    detections.append((x1, y1, x2, y2, confidence))
    
    # Update tracker
    tracked_objects = tracker.update(detections)
    
    # Simplified annotation - skip conveyor belt drawing for speed
    annotated_frame = frame.copy()
    
    egg_sizes = {}
    
    for track_id, bbox in tracked_objects.items():
        x1, y1, x2, y2 = bbox
        
        # Calculate egg size with perspective correction
        width_pixels = x2 - x1
        height_pixels = y2 - y1
        width_cm = width_pixels * horizontal_ratio
        height_cm = height_pixels * vertical_ratio
        egg_sizes[track_id] = (width_cm, height_cm)
        
        # Store size in tracker history - keep only 5 measurements instead of 10
        if track_id in tracker.tracks:
            tracker.tracks[track_id]['size_history'].append((width_cm, height_cm))
            if len(tracker.tracks[track_id]['size_history']) > 5:
                tracker.tracks[track_id]['size_history'] = tracker.tracks[track_id]['size_history'][-5:]
        
        # Simplified visualization - smaller bounding boxes and text
        color = (
            int((track_id * 50) % 255),
            int((track_id * 100) % 255),
            int((track_id * 150) % 255)
        )
        cv.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)  # Thinner lines
        
        # Shorter label for faster rendering
        label = f'ID:{track_id}'
        cv.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)  # Smaller text, thinner font
    
    return annotated_frame, egg_sizes


# read video file and then calculate speed of video(frame per second)
def calculate_video_speed(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    cap.release()
    return fps

def process_video_with_egg_detection_and_tracking(model, input_video_path, output_video_path, conveyor_length_cm=75, camera_height_cm=70):
    """
    OPTIMIZED: Process video with faster egg detection, tracking, and size calculation
    """
    # Open input video
    cap = cv.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {input_video_path}")
    
    # Get video properties
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Read first frame to detect conveyor belt region
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame from video")
    
    # Detect conveyor belt region from first frame
    conveyor_bbox = detect_conveyor_belt_region(first_frame)
    print(f"Detected conveyor belt region: {conveyor_bbox}")
    
    # Calculate perspective-corrected pixel-to-cm ratios
    horizontal_ratio, vertical_ratio, _ = calculate_perspective_corrected_ratio_with_belt_detection(
        camera_height_cm, conveyor_length_cm, first_frame, conveyor_bbox
    )
    
    print(f"Camera setup: {camera_height_cm}cm height, {conveyor_length_cm}cm conveyor length")
    print(f"Pixel ratios - H: {horizontal_ratio:.4f}, V: {vertical_ratio:.4f} cm/px")
    
    # Reset video capture to beginning
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    
    # Initialize tracker with faster settings
    tracker = EggTracker(max_disappeared=3)  # Reduced from 5
    
    # Define codec and create VideoWriter with faster codec
    fourcc = cv.VideoWriter_fourcc(*'XVID')  # Faster than mp4v
    out = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    all_egg_sizes = defaultdict(list)
    
    # Process every frame (or skip frames for even faster processing)
    frame_skip = 1  # Process every frame, set to 2 to process every other frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for faster processing (optional)
        if frame_count % frame_skip != 0:
            out.write(frame)  # Write original frame
            continue
            
        if frame_count % 100 == 0:  # Progress every 100 frames instead of 50
            print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
        
        # Apply optimized egg detection and tracking
        processed_frame, egg_sizes = apply_model_on_frame_with_tracking(
            model, frame, tracker, horizontal_ratio, vertical_ratio, conveyor_bbox
        )
        
        # Store egg sizes for analysis
        for track_id, size in egg_sizes.items():
            all_egg_sizes[track_id].append(size)
        
        # Write the frame to output video
        out.write(processed_frame)
    
    # Release everything
    cap.release()
    out.release()
    
    # Simplified analysis output
    print("\n" + "="*60)
    print("EGG SIZE ANALYSIS (OPTIMIZED)")
    print("="*60)
    
    valid_eggs = 0
    total_avg_width = 0
    total_avg_height = 0
    
    for track_id, sizes in all_egg_sizes.items():
        if len(sizes) > 3:  # Reduced from 5 to 3 frames minimum
            valid_eggs += 1
            avg_width = np.mean([s[0] for s in sizes])
            avg_height = np.mean([s[1] for s in sizes])
            
            total_avg_width += avg_width
            total_avg_height += avg_height
            
            print(f"Egg ID {track_id:2d}: {avg_width:.1f}x{avg_height:.1f}cm | {len(sizes)} frames")
    
    if valid_eggs > 0:
        print("-"*60)
        print(f"Average egg size: {total_avg_width/valid_eggs:.1f} x {total_avg_height/valid_eggs:.1f} cm")
        print(f"Total eggs tracked: {valid_eggs}")
    
    print(f"\nVideo processing completed! Output saved to: {output_video_path}")
    
    return all_egg_sizes

def main():
    model_path = r'model\best.pt'  # Path to your trained model
    video_path = r'input\gettyimages-2202523239-640_adpp.mp4'  # Path to your input video file
    output_video_path = r'output\tracked_eggs_with_sizes.mp4'  # Output video path
    
    # Camera and conveyor parameters
    conveyor_length_cm = 75  # Conveyor length in cm
    camera_height_cm = 70    # Camera height from conveyor in cm
    
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_video_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Load the YOLO model
        print("Loading YOLO model...")
        model = read_model_file(model_path)
        
        # Calculate video speed (FPS)
        fps = calculate_video_speed(video_path)
        print(f"Input video FPS: {fps}")
        
        print(f"Camera setup: {camera_height_cm}cm height, {conveyor_length_cm}cm conveyor length")
        
        # Process video with egg detection, tracking, and size calculation
        print("Starting egg detection, tracking, and size calculation...")
        egg_sizes = process_video_with_egg_detection_and_tracking(
            model, video_path, output_video_path, conveyor_length_cm, camera_height_cm
        )
        
        # Save simplified egg size data to file
        size_report_path = r'output\egg_size_report.txt'
        with open(size_report_path, 'w') as f:
            f.write("EGG SIZE ANALYSIS REPORT (OPTIMIZED)\n")
            f.write("="*50 + "\n")
            f.write(f"Camera Height: {camera_height_cm} cm\n")
            f.write(f"Conveyor Length: {conveyor_length_cm} cm\n")
            f.write("-"*50 + "\n\n")
            
            for track_id, sizes in egg_sizes.items():
                if len(sizes) > 3:  # Reduced threshold from 5 to 3
                    avg_width = np.mean([s[0] for s in sizes])
                    avg_height = np.mean([s[1] for s in sizes])
                    
                    f.write(f"Egg ID {track_id:2d}: {avg_width:.1f}x{avg_height:.1f}cm\n")
        
        print("✅ OPTIMIZED egg tracking completed successfully!")
        print(f"📹 Processed video: {output_video_path}")
        print(f"📊 Size report: {size_report_path}")
        print(f"⚡ Processing optimized for speed!")
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
    except RuntimeError as e:
        print(f"❌ Runtime Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
