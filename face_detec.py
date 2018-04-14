import cv2
import imageio

face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')

def detect(frame):
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces =face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'PERSON', (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        gray_face = gray[y:y+h, x:x+w]
        color_face = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_face, (ex, ey), (ex+w, ey+eh), (0, 255, 0), 2)
    return frame

reader = imageio.get_reader('london.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output-london-text.mp4', fps = fps)
for i, frame in enumerate(reader):
    frame = detect(frame)
    writer.append_data(frame)
    print(i)
writer.close()






