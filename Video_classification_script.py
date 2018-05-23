import numpy as np
import pandas as pd
import keras
import cv2
import os
from keras.preprocessing import image

from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint

def video_to_frames(input_loc, output_loc,frames):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    import cv2
    import os
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (frames-1)):
            break


def move_converted_videos(frames,inpath,filetype):
    classes_str=['Diving-Side', 'Golf-Swing-Back', 'Golf-Swing-Front', 'Golf-Swing-Side', 'Kicking-Front', 'Kicking-Side', 'Lifting', 'Riding-Horse', 'Run-Side', 'SkateBoarding-Front', 'Swing-Bench', 'Swing-SideAngle', 'Walk-Front']
    for classes in classes_str:
        for subdir, dirs, files in os.walk(inpath+classes):
            #print (subdir)
            #print(files)
            if(subdir.endswith("jpeg")):    #to avoid redundancy we dont want another frame folder inside the jpeg folder
                break
            count=0
            for file in files:
                infilepath = subdir + os.sep + file
                outfilepath = subdir + os.sep +"frames"
                #print (infilepath)
                if infilepath.endswith(".avi") and filetype=="avi":
                    video_to_frames(infilepath, outfilepath,frames)
                if infilepath.endswith(".jpg") and filetype=="jpg":
                    #print (infilepath)
                    #print (outfilepath)
                    count=count+1
                    #print(count)
                    move_jpg_to_frames(infilepath, outfilepath,count)
                #print (infilepath,count)

def move_jpg_to_frames(infilepath, output_loc,count):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    imgFile = cv2.imread(infilepath)
    cv2.imwrite(output_loc + "/%#05d.jpg" % (count), imgFile)


def frames_to_dataset(inpath,no_of_frames):
    classes_str=['Diving-Side', 'Golf-Swing-Back', 'Golf-Swing-Front', 'Golf-Swing-Side', 'Kicking-Front', 'Kicking-Side', 'Lifting', 'Riding-Horse', 'Run-Side', 'SkateBoarding-Front', 'Swing-Bench', 'Swing-SideAngle', 'Walk-Front']
    XT=[]
    yt=[]
    count=0
    file_count=0
    for classes in classes_str:
        for subdir, dirs, files in os.walk(inpath+classes):
            
            if(subdir.endswith("frames")):
                print (subdir)
                frame_count=0
                X=[]
                for file in files :
                    frame_count=frame_count+1
                    img_path = subdir + os.sep + file
                    img = image.load_img(img_path, target_size=(256, 256))
                    x = image.img_to_array(img)
                    X.append(x)
                    yt.append(classes)
                    file_count=file_count+1
                    if frame_count==no_of_frames:
                        break
                    #print(file)
                print(frame_count)
                X=np.array(X)
                count=count+1
                XT.append(X)
                
                
                
    return np.array(XT),np.array(yt),count,file_count
    #img = image.load_img(img_path, target_size=(256, 256))
    #x = image.img_to_array(img)

def extract_features(inpath,no_of_frames):
    from keras.applications import VGG16
    conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(256, 256, 3))
    
    Data,target,count,file_count=frames_to_dataset(inpath,no_of_frames)
    print("total number of videos...")
    print(count)
    print("total number of images...")
    print(file_count)
    
    target=target.reshape(file_count,1)
    Data=Data/256
    Data=Data.reshape(file_count,1,256,256,3)
    
    train_feature=[]
    train_labels=[]
    for j in range(target.shape[0]):
        features_batch = conv_base.predict(Data[j])
        train_feature.append(features_batch)
        train_labels.append(target[j])
    
    train_feature=np.array(train_feature)
    train_labels=np.array(train_labels)
    
    train_feature = np.reshape(train_feature, (file_count, 8 * 8 * 512))
    np.save("extracted_features",train_feature)
    np.save("train_labels",train_labels)

def load_data_to_memory(no_of_frames):
    train_features=np.load("extracted_features.npy")
    train_labels=np.load("train_labels.npy")
    X=train_features.reshape(int(train_features.shape[0]/no_of_frames),no_of_frames,train_features.shape[1])
    
    i=0
    y=[]
    while i<int(train_features.shape[0]/no_of_frames):
        y.append(train_labels[i*no_of_frames])
        i=i+1
    y=np.array(y)
    
    from sklearn.preprocessing import LabelEncoder
    encode=LabelEncoder()
    y=encode.fit_transform(y)
    
    from keras.utils import np_utils
    y=np_utils.to_categorical(y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    return X_train, X_test, y_train, y_test

def run_lstm_model(epochs,batch_size):
    X_train, X_test, y_train, y_test=load_data_to_memory(40)
    
    model = Sequential()
    model.add(LSTM(64,dropout=0.2,input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(13, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
    model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=batch_size,nb_epoch=epochs,callbacks=callbacks,shuffle=True,verbose=1)

    return model



#to run the scripts
move_converted_videos(40,"C:\\Users\\mirza914\\Downloads\\MLStuff_3-2\\summer3_2\\video_classification\\ucf_sports_actions\\ucf action\\","jpg")
extract_features("C:\\Users\\mirza914\\Downloads\\MLStuff_3-2\\summer3_2\\video_classification\\ucf_sports_actions\\ucf action\\",40)
run_lstm_model(100,16)
