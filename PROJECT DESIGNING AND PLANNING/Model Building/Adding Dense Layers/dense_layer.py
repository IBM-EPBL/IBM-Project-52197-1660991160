# -*- coding: utf-8 -*-
"""dense_layer

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-QLhbKatukzap_Aq94j1KOopBTMHNlbd
"""

model.add(Dense(units=128,activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 6, activation = 'softmax'))