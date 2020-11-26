from functions import *

df = pd.DataFrame(columns = ['date', 'v_number', 'plate_path' , 'face_path']) #make a pandas datafrmae for csv
df.to_csv('data_new.csv') #save the csv with column names
try:
    os.mkdir("faces")
except FileExistsError:
    print("already exists")
    pass
try:
    os.mkdir("plates")
except FileExistsError:
    print("already exists")
    pass

cap_plate = cv2.VideoCapture("uiet_test.mp4") #path to video
num=[]
while cap_plate.isOpened():
    #time.sleep(0.5)
    ret,frame = cap_plate.read()
    if ret:
        number,plate_img = find_num(frame)
        plate_img = erode(plate_img)
        #cv2.imshow("plate",plate_img) #code to be removed before deployment
        if number:
            num.append(number)
        else:
            continue
        #if num[0] != num[len(num)-1] and len(num)>1:
        if len(num)>30:
            n=most_frequent(num)
            csv_updater(n)
            save_plate(plate_img,n)
            cap_face = cv2.VideoCapture(0)
            ret,face_frame = cap_face.read()
            if ret:
                face = detect(face_frame)
                save_face(face)
                cap_face.release()
            else:
                pass
            print(n)
            num = []
        else:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):  #13 is enter key
            break
    else:
        break
cap_plate.release()