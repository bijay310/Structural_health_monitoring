import cv2
import numpy as np

def main():
    input_path = "input_video.mp4"   # <-- Replace with your video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
   #these are all sample points
    src_points = np.float32([
        [100, 100],  # top-left corner in the frame
        [400, 100],  # top-right corner
        [400, 300],  # bottom-right corner
        [100, 300]   # bottom-left corner
    ])
    
  
    width, height = 300, 200  # Desired size of the rectified output
    dst_points = np.float32([
        [0,      0],       # top-left corner (destination)
        [width,  0],       # top-right corner
        [width,  height],  # bottom-right corner
        [0,      height]   # bottom-left corner
    ])
    
    # Compute the homography
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    
  
    # We use an .avi container with MJPG codec for simplicity.
    output_path = "rectified_output.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Warp the current frame using the homography ---
        rectified_frame = cv2.warpPerspective(frame, H, (width, height))
        
     
        cv2.imshow("Rectified Frame", rectified_frame)
        
        # Write to output file
        out.write(rectified_frame)
        
        # Press 'q' to exit preview early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # clean_all
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
