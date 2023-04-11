import cv2

class Camera(object):
    def __init__(self):
        #capturing video
        self.video = cv2.VideoCapture(1)

    def __del__(self):
        #releasing camera
        self.video.release()

    def get_frame(self):
        #extracting frames
        ret, frame = self.video.read()
        frame = cv2.resize(frame,None,fx=1,fy=1, interpolation=cv2.INTER_AREA) 
        return frame


