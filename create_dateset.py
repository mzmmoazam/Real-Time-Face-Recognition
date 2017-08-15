import cv2
import random
import os
import sys

class preprocess():
    def __init__(self,name):
        self.count = 0
        self.cap = cv2.VideoCapture(0)

        name = './' + name
        if os.path.exists(name):
            print("name already exists")
            name = name + str(random.randint(0, 10000))
            print("So, the dataset has been saved as" + name)

        os.makedirs(name)
        os.chdir(name)

        self.start(name)

        if not os.path.exists(name):
            print("No images exist for the given person")
            sys.exit()

        os.chdir(name)

        print("Creating Proper Dataset.......")
        images_exist = False
        for img in os.listdir("."):
            if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg'):
                self.detect(img, name)
                images_exist = True
        if not images_exist:
            print("No images found to create a dataset")

    def start(self,directory_name):
        while True:
            if cap.grab():
                RET, IMAGE = cap.retrieve()
                if not RET:
                     continue
                global count
                count += 1
                if count % 25 == 0:
                    cv2.imwrite(str(random.uniform(0, 100000)) + ".jpg", IMAGE)
                cv2.imshow("Video", IMAGE)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            count = - 1
        cap.release()
        cv2.destroyAllWindows()

    def detect(self,image_path, name):
        image = cv2.imread(os.path.abspath(image_path))
        # cv2.imshow("Faces Found",image)
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25), flags=0)
        for x, y, w, h in faces:
            sub_img = image[y:y + h, x:x + w]
            os.chdir("./Dataset")
            cv2.imwrite(name + str(random.uniform(0, 100000)) + ".jpg", sub_img)
            os.chdir(name)
            os.chdir(directory_path)

count = 0

cap = cv2.VideoCapture(0)
cascade = "face_cascade.xml"
face_cascade=cv2.CascadeClassifier(cascade)
dir_path = False

def start(directory_name):
    import time
    t0=time.time()
    while (time.time()-t0)/3600 <1:
        print("Time lapsed :",(time.time()-t0)/3600*60)
        if cap.grab():
            RET, IMAGE = cap.retrieve()
            if not RET:
                continue
            global count
            count += 1
            if count % 25 == 0:
                cv2.imwrite(str(random.uniform(0, 100000)) + ".jpg", IMAGE)
            cv2.imshow("Video", IMAGE)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        count = - 1
    cap.release()
    cv2.destroyAllWindows()

def detect(image_path,name):
    image = cv2.imread(os.path.abspath(image_path))
    # cv2.imshow("Faces Found",image)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
    for x,y,w,h in faces:
        sub_img=image[y:y+h,x:x+w]
        os.chdir("../Dataset/")
        print(name+ str(random.uniform(0,100000))+ ".jpg")
        cv2.imwrite(name+ str(random.uniform(0,100000))+ ".jpg",sub_img)
        os.chdir("..")
        os.chdir(name)

def main():
    name = ''
    if len(sys.argv) != 2:
        print("Command line usage : python create_dataset.py <Name of the person>")
        name = input("Enter the name of the person : ")
    else:
        name = sys.argv[1]
      # create label
    name = "./" + name

    if os.path.exists(name):
        print("name already exists")
        name = name + str(random.randint(0, 10000))
        print("So, the dataset has been saved as" + name)

    os.makedirs(name)
    os.chdir(name)

    start(name)
    os.chdir('..')
    if os.path.exists('Dataset'):
        pass
    else:
        os.mkdir('Dataset')

    if not os.path.exists(name):
        print("No images exist for the given person")
        sys.exit()

    os.chdir(name)

    print("Creating Proper Dataset.......")
    images_exist = False
    for img in os.listdir("."):
        if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg'):
            print("detected images")
            detect(img, name)
            images_exist = True
    if not images_exist:
        print("No images found to create a dataset")

if __name__ == "__main__":
    main()
