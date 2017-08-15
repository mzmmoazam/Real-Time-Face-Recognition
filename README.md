# Face-Recognition-
This is face recognition tool that you can train using your webcam (creating your own dataset).

# How to use it 
1. Step one :

Create dataset of images.
```python create_dataset.py name_of_the_person
```
The second argument is name of the person whose dataset has to be made.Don't worry it will ask for a name if you forget to give the command 
line argument.
There will be a pop projecting feed from your webcam.
That feed is actually being recorded , so please don't mind for a minute and give it a good dataset. \
Note : Don't use anything that is circular in your background , cv2 cascade may take it as an eye. (noise)

2. Step two :
 
 ``` python face_recog.py save_model```
 Second argument is if you want to save your model. Default = False
 It will train the images and predict a person in real time.
 

