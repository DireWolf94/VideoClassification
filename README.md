# VideoClassification
An attempt to classify videos using CNN-LSTM network

Each video were split into fixed number of frames and their encoding were extracted using a trained VGG16 model without the top layer.
And a LSTM model was defined to learn the temporal relation between the frames to classify it.
