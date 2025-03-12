import cv2
import mediapipe as mp
import numpy as np
import serial
import tensorflow as tf
import tensorflow_addons as tfa

from PIL import ImageFont, ImageDraw, Image

def myPutText(src, text, pos, font_size, font_color):
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('fonts/gulim.ttc', font_size)
    draw.text(pos, text, font=font, fill=font_color)

    return np.array(img_pil)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

def get_bounding_rectangle(image, detection):
  bounding_box = detection.location_data.relative_bounding_box
  height, width, _ = image.shape # height, width, channel 이미지로부터 세로, 가로 크기 가져옴.

  left = bounding_box.xmin
  top = bounding_box.ymin
  right =  left + bounding_box.width
  bottom = top + bounding_box.height 

  left = (int)(left * width) # left(xmin) 실제 좌표
  top = (int)(top * height) # top(ymin) 실제 좌표
  right = (int)(right * width) # ...
  bottom = (int)(bottom * height) # ...

  if left < 0:
    left = 0

  if top < 0:
    top = 0

  return left, top, right, bottom, width, height

def image_to_numpy():
  image_resize = cv2.resize(crop_image, (TARGET_SIZE, TARGET_SIZE))

  image_s = np.array(image_resize, dtype='float32')
  image_s = image_s.reshape((1, TARGET_SIZE, TARGET_SIZE, 3)) 

  return image_s

TARGET_SIZE = 227
model_path = 'AlexNetMaskDetection.h5'
model = tf.keras.models.load_model(model_path, compile=True)
ser = serial.Serial("COM7", 9600)
COLOR = (255, 255, 255) # 흰색
FONT_SIZE = 30 # 두께

mp_face_detection = mp.solutions.face_detection

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)

    if not success:
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            tmp_high = ser.readline().decode()
            tmp_high = tmp_high[:-1] # 아두이노 통신 or decode 하는동안 끝에 알수 없는 값이 추가되는 것 같다.
            tmp_high = float(tmp_high)

            left, top, right, bottom, width, height = get_bounding_rectangle(image, detection)

            crop_image = image[top:bottom, left:right]
            image_scaled  = image_to_numpy()

            result = model.predict(image_scaled, verbose=0) 
            result = (result[0])[0]

            print(result)

            if result <= 0.5: # 쓴거
              str_mask = "WITH MASK"
              if tmp_high >= 37.5: # 높은거
                color = (50, 50, 200) # R
              else: # 낮은거
                color = (50, 200, 20) # G
            else :  # 안쓴거
              str_mask = "NO MASK"
              color = (50, 50, 200) # R

            cv2.rectangle(image, (left,top), (right, bottom), color, 3, cv2.LINE_4)

            image = myPutText(image, str_mask, (left, top-30), FONT_SIZE, color)
            image = myPutText(image, str(tmp_high)+"℃", (right, top-30), FONT_SIZE, color)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MaskedFaceDetection', cv2.resize(image, None, fx=1.5, fy=1.5))

    if cv2.waitKey(1) == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()