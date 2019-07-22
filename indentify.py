import face_recognition

# Pillow lib to draw the box around the face and for pulling the face out of the image
from PIL import Image, ImageDraw

# Loads the image as Numoy array
image_of_bill = face_recognition.load_image_file('./img/known/Bill Gates.jpg')

# print(image_of_bill)

# It will get the codes for the facial features for the image array
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]
# print(bill_face_encoding)

image_of_steve = face_recognition.load_image_file('./img/known/Steve Jobs.jpg')
steve_face_encoding = face_recognition.face_encodings(image_of_steve)[0]

# Create array of encodings and names
known_face_encodings = [
    bill_face_encoding,
    steve_face_encoding
]

known_face_names = ["Bill Gates",
                    "Steve Jobs"
                    ]
# Load the test image to find faces in

test_image = face_recognition.load_image_file('./img/groups/bill-steve.jpg')

# Find faces in the test image

face_locations = face_recognition.face_locations(test_image)

# get the face encodings for the bill-steve image by passing the face loactions as param
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert the image into PIL format to draw on the image
pil_image = Image.fromarray(test_image)

# Create an ImageDraw instance to draw on the image
draw = ImageDraw.Draw(pil_image)
print(f'Instance of the ImageDraw {draw}')

# Loop through the faces in test image

for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # print(
    #     f'Face encoding as zipped TOP{top}, RIGHT{right}, BOTTOM {bottom}, LEFT{left}')
    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding)

    # The default name
    name = "Unkwon Person"

    # IF there is a match
    if True in matches:
        # get the index where the match happened i.e where it is true
        first_match_index = matches.index(True)

        # Select the name from the name array set on the top
        name = known_face_names[first_match_index]

    #  Draw the Box and Rectangle underneath that with the black strip underneath that
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))

    # Draw a label/text
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10),
                    (right, bottom)), fill=(0, 0, 0), outline=(0, 0, 0))
    draw.text((left + 6, bottom - text_height - 5),
              name, fill=(255, 255, 255, 255))
del draw

# Display the image
pil_image.show()
