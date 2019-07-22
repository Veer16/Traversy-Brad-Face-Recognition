# Pillow library is used to putt the images from a picture
from PIL import Image
import face_recognition

image = face_recognition.load_image_file('./img/groups/team1.jpg')
face_locations = face_recognition.face_locations(image)
print(f'Face locations {face_locations}')

for face_location in face_locations:
    # get coordinates for each face
    top, right, bottom, left = face_location

    face_image = image[top:bottom, left:right]
    print(face_image)
    pil_image = Image.fromarray(face_image)
# To show the images in the image
    pil_image.show()

    # To pull the faces from the image and save the images
    pil_image.save(f'{top}.jpg')
