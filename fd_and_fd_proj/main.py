from face_recognition_module import main_face_recognition
from photo_module import face_landmarks_visualizer
from deleter import delete_subfolders_and_images

num_of_employees = 2
dataset_main_folder = r"C:\Users\franv\OneDrive\Desktop\prova_prog"

# here you create your dataset of faces
for employee in range(num_of_employees):
    face_landmarks_visualizer(save_dir=dataset_main_folder)

main_face_recognition(dataset_main_folder)

delete_subfolders_and_images(dataset_main_folder)


