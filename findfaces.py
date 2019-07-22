import face_recognition

image = face_recognition.load_image_file('./img/groups/team1.jpg')

# Loaction of the faces in the image
face_locations = face_recognition.face_locations(image)

# Array of coordinates of each face
print(face_locations)

# Number of people in the image
print(f'There are {len(face_locations)} people in this image')
