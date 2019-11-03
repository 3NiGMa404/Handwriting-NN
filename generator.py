from keras.preprocessing.image import load_img, img_to_array, array_to_img
import keras
import os
import numpy as np
from keras.layers import Dense, Activation
y=np.ndarray((1,8464))
x=np.ndarray((1,10))

for i in range(10):
    imgs=os.listdir('ds\\'+str(i))
    for img_name in imgs:
        y=np.concatenate((y,np.resize(img_to_array(load_img('ds\\'+str(i)+'\\'+img_name,color_mode="grayscale").resize((92,92))),(1,8464))))
        z=[[0,0,0,0,0,0,0,0,0,0,0]]
        z[0][i]=1
        x=np.concatenate((x,np.resize(np.array(z),(1,10))))
print('starting...')

x=np.delete(x,0,axis=0)

y=np.delete(y,0,axis=0)
model=keras.models.Sequential()
model.add(Dense(96,input_dim=10))
model.add(Activation('relu'))
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dense(192))
model.add(Activation('relu'))
model.add(Dense(192))
model.add(Activation('relu'))
model.add(Dense(384))
model.add(Activation('relu'))
model.add(Dense(384))
model.add(Activation('relu'))
model.add(Dense(768))
model.add(Activation('relu'))
model.add(Dense(900))
model.add(Activation('relu'))
model.add(Dense(1100))
model.add(Activation('relu'))
model.add(Dense(1536))
model.add(Activation('relu'))
model.add(Dense(3072))
model.add(Activation('relu'))
model.add(Dense(8464))
model.compile(loss="mean_squared_error", optimizer='adam',metrics=["accuracy"])
print(x.shape)
print(y.shape)
c=0
while 1:
    c=c+1
    print('Epoch',c)
    try:
        model.fit(x,y,epochs=1,use_multiprocessing=True,workers=8,max_queue_size=30)
    except KeyboardInterrupt:
        break
from PIL import ImageTk,Image
import tkinter
master = tkinter.Tk()
global canvas
canvas = tkinter.Canvas(master, width = 92, height = 92)  
canvas.pack()
w1 = tkinter.Scale(master, from_=0, to=255,resolution=0.1,length=200)
w1.pack()
var = tkinter.IntVar()
c = tkinter.Checkbutton(master, text="High Contrast",variable=var)
c.pack()
global img
img = ImageTk.PhotoImage(Image.open("_3.png"))
def show_values():
      
    tot=0
    for i in range(10):
        z=[0,0,0,0,0,0,0,0,0,0]
        z[i]=1
        arr=np.resize(model.predict(np.resize(np.array(z),(1,10))),(92,92))
        prearr=arr
        print(var.get())
        if var.get():
            arr[arr >= w1.get()] = 255
            arr[arr < w1.get()] = 0
        arr=np.resize(arr,(92,92,1))
        arr=np.concatenate((arr,arr,arr),axis=2)
        array_to_img(arr).save('_'+str(i)+'.png')
        tot=tot+np.average(prearr)
    print('Average darkness: '+str(tot/10))
    img = ImageTk.PhotoImage(Image.open("_3.png")) 
    canvas.create_image(20,20, anchor=tkinter.NW, image=img)    
    canvas.image = img   

tkinter.Button(master, text='Go', command=show_values).pack()

tkinter.mainloop()
'''
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''
