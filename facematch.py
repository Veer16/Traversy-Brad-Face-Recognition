import face_recognition

# Load image as Numpy array
image_of_bill = face_recognition.load_image_file('./img/known/Bill Gates.jpg')

# It will get the codes for the facial features for the image array
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]
print(f'Face encodind {bill_face_encoding}')
unkown_image = face_recognition.load_image_file(
    './img/unknown/bill-gates-4.jpg')
unkown_face_encoding = face_recognition.face_encodings(unkown_image)[0]

# compare faces

results = face_recognition.compare_faces(
    [bill_face_encoding], unkown_face_encoding)
print(results)
if results[0]:

    print('This is Bill Gates')
else:
    print('This is NOT Bill Gates')
