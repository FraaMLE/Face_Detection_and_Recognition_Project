import os

def delete_subfolders_and_images(main_folder):
    for root, dirs, files in os.walk(main_folder, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)  # Delete the file
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)  # Delete the subfolder
if __name__ == "__main__":
    pass
# # Example usage
# main_folder = r"C:\Users\franv\OneDrive\Desktop\prova_prog"
# delete_subfolders_and_images(main_folder)