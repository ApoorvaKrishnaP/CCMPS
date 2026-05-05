import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "dense_4.mp4"

CONF = 0.10
IOU = 0.5

TILE_SIZE = 640
OVERLAP = 0.2
PROCESS_EVERY_N_FRAMES = 8

LOW_THRESHOLD = 10
HIGH_THRESHOLD = 25

model = YOLO(MODEL_PATH)

# -------- DETECTION --------
def nms(boxes, scores):
    if len(boxes) == 0:
        return []
    boxes_xywh = [[x1, y1, x2-x1, y2-y1] for x1,y1,x2,y2 in boxes]
    idx = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.0, 0.4)
    return [boxes[i] for i in idx.flatten()] if len(idx)>0 else []

def tile_detect(frame):
    h,w,_ = frame.shape
    step = int(TILE_SIZE*(1-OVERLAP))
    boxes, scores = [], []

    for y in range(0,h,step):
        for x in range(0,w,step):
            tile = frame[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if tile.shape[0]<100 or tile.shape[1]<100:
                continue

            res = model(tile, conf=CONF, iou=IOU, verbose=False)

            for r in res:
                if r.boxes is None: continue
                for b in r.boxes:
                    if int(b.cls[0])!=0: continue
                    x1,y1,x2,y2 = map(int,b.xyxy[0])
                    boxes.append([x1+x,y1+y,x2+x,y2+y])
                    scores.append(float(b.conf[0]))
    return boxes, scores

# -------- SPLIT VIDEO --------
def split_view(frame, boxes):
    h,w = frame.shape[:2]
    zh,zw = h//2,w//3
    out = np.zeros_like(frame)
    zid=1

    for r in range(2):
        for c in range(3):
            y1,y2 = r*zh,(r+1)*zh
            x1,x2 = c*zw,(c+1)*zw
            zone = frame[y1:y2,x1:x2].copy()

            count=0
            for (bx1,by1,bx2,by2) in boxes:
                cx,cy=(bx1+bx2)//2,(by1+by2)//2
                if x1<=cx<=x2 and y1<=cy<=y2:
                    count+=1
                    cv2.rectangle(zone,(bx1-x1,by1-y1),(bx2-x1,by2-y1),(0,255,0),2)

            if count<LOW_THRESHOLD:
                col=(0,255,0); lab="LOW"
            elif count<HIGH_THRESHOLD:
                col=(0,165,255); lab="MEDIUM"
            else:
                col=(0,0,255); lab="HIGH"

            cv2.rectangle(zone,(0,0),(zw,zh),col,3)
            cv2.putText(zone,f"Z{zid}:{count} ({lab})",(10,30),0,0.7,col,2)

            out[y1:y2,x1:x2]=zone
            zid+=1

    cv2.putText(out,f"TOTAL: {len(boxes)}",(40,80),0,2,(0,0,255),4)
    return out

# -------- ZONE TWIN --------
def zone_twin(boxes, shape):
    h,w = shape[:2]
    zh,zw = h//2,w//3
    twin = np.zeros((300,600,3),dtype=np.uint8)

    for i in range(6):
        r,c=i//3,i%3
        y1,y2=r*zh,(r+1)*zh
        x1,x2=c*zw,(c+1)*zw

        cnt=0
        for (bx1,by1,bx2,by2) in boxes:
            cx,cy=(bx1+bx2)//2,(by1+by2)//2
            if x1<=cx<=x2 and y1<=cy<=y2:
                cnt+=1

        if cnt<LOW_THRESHOLD: col=(0,255,0)
        elif cnt<HIGH_THRESHOLD: col=(0,165,255)
        else: col=(0,0,255)

        cv2.rectangle(twin,(c*200,r*150),(c*200+200,r*150+150),col,-1)
        cv2.rectangle(twin,(c*200,r*150),(c*200+200,r*150+150),(255,255,255),2)
        cv2.putText(twin,f"Z{i+1}:{cnt}",(c*200+10,r*150+70),0,1,(0,0,0),2)

    return twin

# -------- DOT TWIN --------
def dot_twin(boxes, shape):
    h,w=shape[:2]
    grid=np.zeros((20,20))
    ch,cw=h//20,w//20

    for (x1,y1,x2,y2) in boxes:
        cx,cy=(x1+x2)//2,(y1+y2)//2
        grid[min(cy//ch,19)][min(cx//cw,19)]+=1

    vis=np.zeros((500,500,3),dtype=np.uint8)

    for r in range(20):
        for c in range(20):
            v=grid[r][c]
            if v==0: col=(120,120,120)
            elif v<2: col=(0,255,0)
            elif v<4: col=(0,165,255)
            else: col=(0,0,255)

            cv2.rectangle(vis,(c*25,r*25),(c*25+25,r*25+25),col,-1)
            cv2.rectangle(vis,(c*25,r*25),(c*25+25,r*25+25),(0,0,0),1)

    return vis

# -------- GRAPH --------
history = []

def draw_graph():
    h,w = 200,600
    img = np.zeros((h,w,3),dtype=np.uint8)

    if len(history)>1:
        maxv = max(history)+1
        for i in range(1,len(history)):
            x1 = int((i-1)/len(history)*w)
            x2 = int(i/len(history)*w)
            y1 = h - int(history[i-1]/maxv*h)
            y2 = h - int(history[i]/maxv*h)
            cv2.line(img,(x1,y1),(x2,y2),(0,255,255),2)

    return img

# -------- WINDOWS --------
class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video")
        self.label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analytics")

        self.zone = QLabel()
        self.dot = QLabel()
        self.graph = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.zone)
        layout.addWidget(self.dot)
        layout.addWidget(self.graph)
        self.setLayout(layout)

# -------- MAIN --------
class Controller:
    def __init__(self, vw, db):
        self.vw = vw
        self.db = db
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.frame_id=0
        self.prev=[]

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        self.frame_id+=1
        frame=cv2.resize(frame,(1280,720))

        if self.frame_id%PROCESS_EVERY_N_FRAMES==0:
            b,s=tile_detect(frame)
            self.prev=nms(b,s)
        else:
            b=self.prev

        history.append(len(b))
        if len(history)>100: history.pop(0)

        vid = split_view(frame,b)
        zone = zone_twin(b,frame.shape)
        dot = dot_twin(b,frame.shape)
        graph = draw_graph()

        self.vw.label.setPixmap(self.to_qt(vid))
        self.db.zone.setPixmap(self.to_qt(zone))
        self.db.dot.setPixmap(self.to_qt(dot))
        self.db.graph.setPixmap(self.to_qt(graph))

    def to_qt(self,img):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,ch=img.shape
        return QPixmap.fromImage(QImage(img.data,w,h,ch*w,QImage.Format_RGB888))


# -------- RUN --------
app = QApplication(sys.argv)

vw = VideoWindow()
db = Dashboard()

ctrl = Controller(vw, db)

vw.show()
db.show()

sys.exit(app.exec_())
