from functions import *


cap_plate = cv2.VideoCapture("uiet_test.mp4")
num=[]
while cap_plate.isOpened():
    #time.sleep(0.05)
    ret,frame = cap_plate.read()
    if ret:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        try:
            vehicle, LpImg,cor = get_plate(frame)
            plate = draw_box(frame,cor)
            #plate = image_for_ocr(LpImg)[2]
        except AssertionError:
            plate = frame
        cv2.imshow("plate",plate)
        if cv2.waitKey(1) & 0xFF == ord('q'):  #13 is enter key
            break        
    else:
        break
cap_plate.release()
cv2.destroyAllWindows()