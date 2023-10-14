import cv2
import os
import cvzone
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture("Resources/Videos/vddo.mp4")
#shirtlist
shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)
fixedRation = 432/190
shirtRatioHeightWidth = 577/432
while True:
    success, img = cap.read()
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Extract specific landmarks for positions
        landmarks = results.pose_landmarks.landmark

        # Example: Find the position of the left hand (landmark 9)
        lm11 = landmarks[11]
        lm11 = [int(lm11.x * img.shape[1]), int(lm11.y * img.shape[0])]
        lm12 = landmarks[12]
        lm12 = [int(lm12.x * img.shape[1]), int(lm12.y * img.shape[0])]
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[0]), cv2.IMREAD_UNCHANGED)

        widthOfShirt = int((lm11[0] - lm12[0]) * fixedRation)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
        currentScale = (lm11[0] - lm12[0]) / 190
        offset = int(128 * currentScale), int(140 * currentScale)
        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0]-offset[0], lm12[1]-offset[1]))
        except:
            pass
    cv2.imshow("Pose Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # Press the 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()
