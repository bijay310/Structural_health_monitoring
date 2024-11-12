import cv2
import numpy as np
import matplotlib.pyplot as plt

# Camera calibration parameters (from checkerboard_calibration.py)
cameraMatrix = np.array([
    [1.13701650e+03, 0.00000000e+00, 9.39024055e+02],
    [0.00000000e+00, 1.12147552e+03, 3.23427161e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
distCoeffs = np.array([[-0.35468426, 0.27247466, 0.01016477, -0.01181304, -0.14494492]])

#I am using results from first r and t vecs. We can take median for all the 
#results and feed that for somewhat improved accuracy. I tried that but didn't 
#find any improvements
rvec = np.array([[ 0.12013852],
       [-0.94866831],
       [-1.23951908]])
tvec = np.array([[-15.47781493],
       [  0.40229841],
       [ 19.03291815]])


video_path = "C:\\Users\\bijay\\Desktop\\5th Sem\\Dr. Marin Research\\Experiment_videos\\exp_1_side_clipped_2.mp4"
cap = cv2.VideoCapture(video_path)

# Click event to select the initial point
def select_point(event, x, y, flags, param):
    global target_point, point_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        target_point = np.array([[x, y]], dtype=np.float32)
        point_selected = True
        print(f"Point selected at: {x}, {y}")

# Initialize variables
point_selected = False
target_point = None
displacement_mm = []
frame_count = 0

# Read the first frame and select the target point
ret, first_frame = cap.read()
if not ret:
    print("Error: Unable to read video")
    cap.release()
    exit()

# Display the first frame and wait for point selection
cv2.namedWindow("Select Point", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select Point", select_point)

while not point_selected:
    # Show the first frame and display a message
    cv2.imshow("Select Point", first_frame)
    cv2.putText(first_frame, 
                "Click to select a point", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_AA)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

if not point_selected:
    print("No point selected. Exiting.")
    cap.release()
    exit()

# Draw marker on selected point for verification
cv2.circle(first_frame, (int(target_point[0][0]), int(target_point[0][1])), 5, (0, 255, 0), -1)

undistorted_point = cv2.undistortPoints(target_point, cameraMatrix, distCoeffs)

z_depth = tvec[2, 0]
normalized_point = undistorted_point[0][0] * z_depth

initial_target_3D = np.array([
    [normalized_point[0]],
    [normalized_point[1]],
    [z_depth]
])

# Loop over each frame to track the point and calculate displacement
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Track the point using optical flow
    new_point, st, _ = cv2.calcOpticalFlowPyrLK(first_frame, frame, target_point, None)
    
    # Ensure point was found and is within frame bounds
    if new_point is not None and st[0][0] == 1:
        # Undistort the new point and convert to normalized camera coordinates
        undistorted_new_point = cv2.undistortPoints(new_point, cameraMatrix, distCoeffs)
        normalized_new_point = undistorted_new_point[0][0] * z_depth
        
        # Calculate the new 3D coordinates
        new_target_3D = np.array([
            [normalized_new_point[0]],
            [normalized_new_point[1]],
            [z_depth]
        ])
        
        # Calculate displacement in mm
        frame_displacement_mm = np.linalg.norm(new_target_3D - initial_target_3D)
        displacement_mm.append(frame_displacement_mm)

        # Update initial frame and target point for next iteration
        target_point = new_point
        first_frame = frame.copy()
        
    frame_count += 1

cap.release()

# Plot the displacement over time
plt.figure(figsize=(10, 6))
plt.plot(displacement_mm, label="Displacement (mm)")
plt.xlabel("Frame")
plt.ylabel("Displacement (mm)")
plt.title("Point Displacement over Time")
plt.legend()
plt.show()
