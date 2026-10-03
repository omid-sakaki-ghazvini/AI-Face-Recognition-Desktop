import os


path = r"D:\face_recognition\pythonProject\dataset\sakaki\1.jpg"


print("Exists:")
print(os.path.exists(path))


print("\nFolder files:")

folder = r"D:\face_recognition\pythonProject\dataset\sakaki"

if os.path.exists(folder):

    for f in os.listdir(folder):
        print(f)

else:

    print("Folder not found")