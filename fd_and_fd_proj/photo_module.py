import os
import cv2
import mediapipe as mp

def capture_and_save_image(frame, save_dir, name):
    person_dir = os.path.join(save_dir, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    image_count = len(os.listdir(person_dir))
    image_path = os.path.join(person_dir, f"captured_image_{image_count + 1}.jpg")
    
    cv2.imwrite(image_path, frame)
    print(f"Image captured and saved at {image_path}")

def face_landmarks_visualizer(save_dir="captured_images"):
    name = input("Enter the name of the person: ")
    if not name:
        print("Name cannot be empty.")
        return
    
    CUSTOM_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [5, 6], [6, 7], [7, 8], [8, 9],
        [10, 11], [11, 12], [12, 13], [13, 14],
        [15, 16], [16, 17], [17, 18], [18, 19],
        # Define your own connections as needed
    ]

    cap = cv2.VideoCapture(0)

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    while True:
        success, img = cap.read()
        to_store = img.copy()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                for connection in CUSTOM_CONNECTIONS:
                    x0, y0 = int(faceLms.landmark[connection[0]].x * iw), int(faceLms.landmark[connection[0]].y * ih)
                    x1, y1 = int(faceLms.landmark[connection[1]].x * iw), int(faceLms.landmark[connection[1]].y * ih)
                    cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 1)

        cv2.putText(img, "Instructions: Press 't' to capture and save an image.", (20, img.shape[0] - 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(img, "Take one frontal pic, and two looking diagonally.", (20, img.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('t'):
            capture_and_save_image(to_store, save_dir, name)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pass