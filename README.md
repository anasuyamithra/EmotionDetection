# EmotionDetection
Realtime Emotion Detection Using The Keras Framework

No aspect of our mental life is more important to the quality and meaning of our existence than the emotions. And being able to gauge the emotion of a subject using a self learning algorithm has been hotly chased for by several billion dollar fortune 500 companies for the past two decades. The real world applications of this technology include helping law enforcement, crowd control, advertising, etc. Emotion being a subjective thing, employing knowledge and science behind a set of labeled data and extracting the components that constitute it,is a challenging proposition. Gestures, actions, postures, behaviour, face jargon and vocalizations; these are painstaking as a standard that convey emotions of individual beings. 
Expansive research has been finished to explore the associations between these channels and sentiments. 

![readme_img1](https://user-images.githubusercontent.com/60477228/107118687-69937d00-68a8-11eb-99fe-31c856b61ce2.png)

In computer science, affective computing is a branch of the study and development of artificial intelligence that deals with the design of systems and devices that can recognize, interpret, and process human emotions.

The more modern branch of computer science originated with Rosalind Picard's 1995 paper on affective computing, and the field of computer assisted emotion detection has come a long way from those days. In the age of 2020, we have several libraries and frameworks that can assist almost anyone with decent computer science knowledge, in making their own program for real time emotion detection. Some of these frameworks include opencv2, tensorflow, etc.

In this project we will be using the Keras framework to perform realtime emotion analysis.

![readme_img2](https://user-images.githubusercontent.com/60477228/107118684-68625000-68a8-11eb-87ba-ca232235aa46.png)

Keras contains numerous implementations of commonly used neural-network building blocks such as layers, objectives, activation functions, optimizers, and a host of tools to make working with image and text data easier to simplify the coding necessary for writing deep neural network code.

Project Details:
  For the project we will be using the following libraries and frameworks:
Keras
Tensorflow
Numpy
Pandas
OpenCV2

![readme_img3](https://user-images.githubusercontent.com/60477228/107118696-6d270400-68a8-11eb-869f-fb755d564f62.png)

The framework we’ll be using for the most part will be Keras. Using the Kaggle dataset and Haar cascades XML file.

The haar cascade xml file we are going to use is the “haarcascade full frontal face default”.
We will use the above xml file because the input we are providing it is the full frontal face of the subject through a webcam.

The Kaggle data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

The training csv file contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. The training csv provided in the kaggle date set only contains the pixel column, and our program will be responsible for providing the emotion column.



The training set consists of 28,709 examples, which our program will use. We’ll run 100-150 epochs of the training set in order to minimise the error rate as much as possible. All the training data will be tabulated using the pandas library function. The expected training time for the above program is 4-5 hours on our Legion Y-540 laptop with an i7 9th gen processor and GTX 1660ti graphics processor. 

The program will take in a live webcam feed of our subject and will, in realtime, output the emotion he or she is showing at that very instant. In order to take in this webcam feed, we will be using an opencv2 library function.

# Output:
## Main Window-
![Test Image 4](https://github.com/akaashnidhiss/emotion_detection/blob/main/main_window.png?raw=true)
## Save to root directory functionality-
![Test Image 4](https://github.com/akaashnidhiss/emotion_detection/blob/main/save_webcam_image.png?raw=true)
## Example of saved screenshot-
![Test Image 4](https://github.com/akaashnidhiss/emotion_detection/blob/main/example_screenshot.png?raw=true)


Project partner: github.com/akaashnidhiss
