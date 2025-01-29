import cv2
import numpy as np

# Load video
video_path = "C:\\Users\\bijay\\Desktop\\5th Sem\\Dr. Marin Research\\Experiment_videos\\Shortened Experiment 1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video writer
out = cv2.VideoWriter('rectified_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define source points (from the original distorted view)
src_pts = np.array([[50, 100], [400, 100], [50, 300], [400, 300]], dtype=np.float32)

# Define destination points (desired rectified view)
dst_pts = np.array([[0, 0], [frame_width, 0], [0, frame_height], [frame_width, frame_height]], dtype=np.float32)

# Compute Homography matrix
H, _ = cv2.findHomography(src_pts, dst_pts)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply homography transformation
    rectified_frame = cv2.warpPerspective(frame, H, (frame_width, frame_height))

    # Show rectified frame (optional)
    cv2.imshow('Rectified Frame', rectified_frame)

    # Write frame to output video
    out.write(rectified_frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
