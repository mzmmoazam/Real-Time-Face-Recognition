import os, shelve,numpy as np,cv2,re,sys
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import  shuffle

Datafile = shelve.open("Data.db")
if 'Data' not in Datafile.keys():
    Datafile['Data'] = list()
    Data_list = list()
else:
    Data_list = Datafile["Data"]


def Make_Changes(label):
    if label not in Data_list:
        Data_list.append(label)
        print(Data_list)




def get_images(path):

    images = np.ndarray(shape=(len(os.listdir(path)), 48, 48, 3),
                        dtype=np.float32)
    labels = list()
    count = -1
    if len(os.listdir(path)) == 0:
        print("Empty Dataset.......aborting Training")
        exit()
    for img in os.listdir(path):
        regex = re.compile(r'(\d+|\s+)')
        labl = regex.split(img)
        labl = labl[0]
        count = count + 1
        Make_Changes(labl)
        image_path = os.path.join(path, img)
        image = cv2.imread(image_path)
        try:
            image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print("ye lo error", e)
            exit()


        if images[count, :, :].shape != image.shape:
            print(image.shape)
            count -= 1
            continue
        images[count, :, :] = image
        labels.append(Data_list.index(labl)) # one hot encoding here
    images,labels = shuffle(images,labels)
    return images, labels, count


def net(X, Y,save_model=False):
    tflearn.config.init_graph(gpu_memory_fraction=1)
    
    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    
    # Building convolutional network
    network = input_data(shape=[None, 48, 48, 3], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 2, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 2, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 328, activation='relu')
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = fully_connected(network, 128, activation='relu')
    # network = dropout(network, 0.8)
    network = fully_connected(network,len(set(Y))+1, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='softmax_categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=3)
    # print(Y)
    print(np.eye(len(set(Y))+1)[Y])
    model.fit({'input': X}, {'target': np.array(np.eye(len(set(Y))+1)[Y])}, n_epoch=15,batch_size=50,
              validation_set=0.3,

              snapshot_step=15000, show_metric=True, run_id='face_recogn')
    if os.path.exists('fr.tflearn.index'):
        print('Loading pre-trained model')
        model.load('fr.tflearn')
    if save_model:
        model.save('fr.tflearn')
    return model



def initialize_recognizer(save_model=False):
    # You can use cv2 face classifier too ,if so uncomment the line below, train the classifier and return it ,
    # some minor changes to input data have to be done
    # like greyscale etc..

    # try:
    #     face_recognizer = cv2.face.createLBPHFaceRecognizer()
    # except:
    #     face_recognizer = cv2.createLBPHFaceRecognizer()

    print("Training..........")
    Dataset = get_images("./Dataset/")
    print("Recognizer trained using Dataset: " + str(Dataset[2]) + " Images used")
    # print(Dataset[0])
    return net(Dataset[0], Dataset[1],save_model)



FONT = cv2.FONT_HERSHEY_SIMPLEX
CASCADE = "face_cascade.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE)

def recognize_video(face_recognizer):
    cap = cv2.VideoCapture(0)
    test_img=np.ndarray(shape=(1, 48, 48, 3), dtype=np.float32)
    while True:
        if cap.grab():
            ref, image = cap.retrieve()
            image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25),
                                                  flags=0)
            for x, y, w, h in faces:
                sub_img = image_grey[y:y + h, x:x + w]
                img = image[y:y + h, x:x + w]
                try:
                    img = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
                except Exception as e:
                    print(e)
                    continue
                test_img[0,:,:] = img
                ans = face_recognizer.predict(test_img)
                name = ''
                try:
                    print(ans,[round(i)for i in ans[0]])
                    name = Data_list[list(ans[0]).index(1)]
                except Exception as e:
                    print(e)
                    ans = [round(i)for i in ans[0]]
                    name = Data_list[list(ans).index(1)]
                cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 255, 0), 2)
                cv2.putText(image, name, (x, y - 10), FONT, 0.5, (255, 255, 0), 1)
            cv2.imshow("Faces Found", image)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1) & 0xFF == ord('Q')):
                break
    Datafile["Data"] = Data_list
    Datafile.close()
    cap.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    save_model = False
    if len(sys.argv) == 2:
        if sys.argv[2] == 'save':
            save_model = True

    face_r = initialize_recognizer(save_model)
    recognize_video(face_r)
