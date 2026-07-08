import cv2

url = "rtsp://admin:MeYyPa@10.96.188.165:8556"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

print("Opened:", cap.isOpened())

ret, frame = cap.read()
print("Read frame:", ret)

cap.release()