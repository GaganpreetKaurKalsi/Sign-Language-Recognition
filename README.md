# Sign-Language-Recognition Project

<div align="center"><img src="./ASL.png" style="height: 300px;"></div>
<br>

## Introduction

A project that will help you translate american sign language to english alphabets. I have developed 2 interfaces :- detection in realtime and an application deployed using Flask where you can either upload an image of a sign language or you can click photos and then predict.

## Dataset Used

The novelity of the project is it's dataset. The dataset used for training the project is a self developed one. I used mediapipe to create the same. 

**Idea behind developing the dataset** - Many of you must have heard of or some might have also drawn stick figures in their drawing classes. For the ones who are unaware of the term - *Stick figures are drawings of humans where we use lines to draw the human figure.* They are the most basic when it comes to drawing human structure. We draw humans in different poses using sticks. Mediapipe is the same. It detects our hand and draws it in terms of dots and lines which are easy to comprehend. According to me instead of using hands where there are lost of variations like color, size, space covered; it is much more effective if we train our model on hands drawn with dots and lines.

I have got a very positive output with this dataset which was quite unexpected. The dataset creation was full of DIP(Digital Image Processing). The dataset will follow improvements in future.

<br>

👇👇 **Below are some pics of the dataset** 👇👇

<br>

<img src="./SignLanguageDataset.png" style="height: 450px;">

## Limitations
- Detecting only right hand
- Covering A-Z alphabets
