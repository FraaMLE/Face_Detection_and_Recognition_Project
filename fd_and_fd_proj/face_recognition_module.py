import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os


model = InceptionResnetV1(pretrained='vggface2').eval()
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

def face_coordinates(detection,img):
    bboxC = detection.location_data.relative_bounding_box
    # get the unnormalized face coordinates
    ih, iw, ic = img.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
    cropped_face = img[y:y+h, x:x+w]
    bbox = x, y, w, h
    return cropped_face, bbox

def from_path_to_det(img_path):
    img = cv2.imread(img_path)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    return img,imgRGB,results

def cropper(img_path):
    img,imgRGB,results = from_path_to_det(img_path)
    if results.detections:
        for id, detection in enumerate(results.detections):
            face_img, bbox = face_coordinates(detection,img)
            if not face_img.size == 0:  # Check if the cropped face image is not empty
                return cv2.resize(face_img, (100, 100))
    # No face detected or face_img is empty
    return cv2.resize(imgRGB, (28, 28))

def numpy_to_pil(numpy_image):
    return Image.fromarray((numpy_image).astype('uint8'))

def embedding_getter(face):
    emb = transform(face).unsqueeze(0)
    with torch.no_grad():
        # it has to be RGB
        embeddings = model(emb)  # Pass the processed face tensor to the model
    
    return embeddings

transform = transforms.Compose([
    transforms.Lambda(numpy_to_pil),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def calculate_euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def drawer(label, bbox,img):
    import cv2

    if label == "Unknown":
        rect_color = (0, 0, 255)  # red
    else:
        rect_color = (0, 255, 0)  # green

    stripe_height = 25
    cv2.rectangle(img, (bbox[0], bbox[1] - stripe_height), (bbox[0] + bbox[2], bbox[1]), rect_color, -1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2  # Increase this value for a thicker text
    
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = bbox[0]
    text_y = bbox[1] - 5
    
    cv2.rectangle(img, (text_x, text_y + 5), (text_x + text_size[0], text_y - text_size[1]), rect_color, -1)
    cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    cv2.rectangle(img, bbox, rect_color, 2)

def get_image_paths(main_folder, extensions=['.jpg', '.png']):
    image_paths = []

    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


def get_subfolder_name(image_path):
    subfolder_name = os.path.basename(os.path.dirname(image_path))
    return subfolder_name


def dataset_returner(paths):
    dataset = [(embedding_getter(cropper(i)), get_subfolder_name(i)) for i in paths]
    return dataset

paths = get_image_paths(r"C:\Users\franv\OneDrive\Desktop\cropped_dataset")
dataset = dataset_returner(paths)


def find_matching_embedding(new_embedding, dataset, threshold = 0.7):
    best_match_index = -1
    min_distance = float('inf')

    for idx, (db_embedding, label) in enumerate(dataset):
        distance = calculate_euclidean_distance(new_embedding, db_embedding)
        if distance < min_distance:
            min_distance = distance
            best_match_index = idx

    if min_distance <= threshold:
        return dataset[best_match_index][1]  # Return the label from the dataset tuple
    else:
        return "Unknown"


def main_face_recognition(folder_path):
    paths = get_image_paths(folder_path)
    dataset = dataset_returner(paths)

    cap = cv2.VideoCapture(0)
    counter = 0
    etichette = []
    list_of_faces = []
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)

        if counter % 30 == 0 or counter == 0:
            etichette.clear()
            if results.detections:
                for id, detection in enumerate(results.detections):
                    face_img, bbox = face_coordinates(detection,img)
                    try:
                        new_embedding = embedding_getter(face_img)
                    except:
                        cv2.rectangle(img, bbox, (255,255, 255), 2)
                    else:
                        label = find_matching_embedding(new_embedding, dataset)
                        etichette.append(label)
                        drawer(label,bbox,img)

        else:
            if results.detections:
                for id, detection in enumerate(results.detections):
                    face_img, bbox = face_coordinates(detection,img)
                    list_of_faces.append(bbox)

                if len(etichette) >= len(list_of_faces):
                    for faccia in range(len(list_of_faces)):
                        if faccia < len(etichette):  # Ensure the index is within the range of etichette
                            drawer(etichette[faccia],list_of_faces[faccia],img)
            list_of_faces.clear()

        counter += 1
        cv2.imshow('Face Recognition going on', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pass