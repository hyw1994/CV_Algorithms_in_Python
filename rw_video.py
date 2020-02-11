import cv2
import os
import time
# read a video
data_path = os.path.join(os.getcwd(), "data")
video_path = os.path.join(data_path, "plant.avi")
cap = cv2.VideoCapture(video_path)

print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(video_fps)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(data_path, "gray_video.mp4"),fourcc, int(video_fps), (int(video_width), int(video_height)), isColor=False)

# play the video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # if we want, we can change the frame to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    out.write(gray)
    if cv2.waitKey(int(1000/video_fps)) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Record the video from camera
cap = cv2.VideoCapture(0)
# fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Number of frames to capture
num_frames = 120;
    
print("Capturing {0} frames".format(num_frames))

# Start time
start = time.time()
    
# Grab a few frames
for i in range(num_frames) :
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    
# End time
end = time.time()

# Time elapsed
seconds = end - start
print("Time taken : {0} seconds".format(seconds))

# Calculate frames per second
fps  = num_frames / seconds;
print("Estimated frames per second : {0}".format(fps));



# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(data_path, "camera.mp4") ,fourcc, int(fps), (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    if ret==True:

        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
out.release()

cv2.destroyAllWindows()