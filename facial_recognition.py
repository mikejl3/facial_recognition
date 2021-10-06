
import os
import cv2
import numpy as np
import glob
import face_recognition
import webbrowser


faces_encodings = []
faces_names = []
cur_direc = os.getcwd()
# section1
path = 'C:\\Users\\mikej\\PycharmProjects\\facial_recognition\\faces\\'

list_of_files = [f for f in glob.glob(path+'*.jpg')]

number_files = len(list_of_files)

names = list_of_files.copy()

# section2
for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    faces_encodings.append(globals()['image_encoding_{}'.format(i)])
# Create array of known names
    names[i] = names[i].replace(cur_direc, "")
    names[i] = names[i].replace('\\faces\\', "")
    names[i] = names[i].replace('-', ' ')
    names[i] = names[i].replace('.JPG', '')
    faces_names.append(names[i])


# section3
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# section4
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    fgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(fgb_small_frame)
        face_encodings = face_recognition.face_encodings(fgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(faces_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]
                if name == "Cullen Wallace":
                    webbrowser.open(
                        'http://lh5.ggpht.com/-4zQKpevpRLU/T8v9ZAdx04I/AAAAAAAAKik/8uHs2lAm0_k/david_thumb%25255B7%25255D.jpg?imgmax=800',
                        new=1)
                    exit()
            face_names.append(name)
    process_this_frame = not process_this_frame


    # displaying the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
# draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

# input text label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# display the resulting image
    cv2.imshow('Video', frame)


    # quit by hitting q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

