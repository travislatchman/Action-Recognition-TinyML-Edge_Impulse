# Action-Recognition-TinyML-Edge_Impulse



## Gesture Recognition Training a TensorFlow Lite Micro Model For Arduino

Gesture Recognition Training a TensorFlow Lite Micro Model For Arduino where a custom model is trained
on data captured from an Arduino Nano 33 BLE Sense.

### i. Gesture Selection & Data Recording
Two common kitchen tasks were used as Chosen gestures. selected actions
attach the nano-sense BLE to your hand and record several instances of the actio
- Choose two gestures commonly used in cooking. Examples include cutting a tomato, peeling a potato, pouring water into a pot, etc.
- Attach the nano-sense BLE to your hand and record several instances of these actions.
- Additionally, record these actions with your phone's camera.
- Name your data files as `ActionName_YourLastNameFirstNamelnitial.csv`. For instance, my peeling potato data would be named `PeelingPotato_AndreouA.csv`.
- Upload the collected data to the shared data directory. 



### ii. Data Visualization
capturing the data by visualizing the data in the Arduino device. 
- Ensure your data is being captured correctly by visualizing it in the Arduino device.
- Capture a screenshot of your visualization window.



### iii. Model Training in TensorFlow

Train your model using TensorFlow.
Upload model to Arduino device and do inference on device displaying result on a
screen



### iv. Inference on Arduino Device

- Upload the trained model to your Arduino device.
- Perform inference directly on the device and display the results on a screen.


### v. Data Quantity Experimentation

Experiment by increasing or decreasing the number of recordings per gesture. Analyze and note how this variation impacts the performance.



### vi. Data Source Experimentation

Experiment with using only the accelerometer or gyroscope data (not both). Analyze and determine the impact on performance.

increasing and decreasing the number of recordings per gesture, how does this
impact performance?

Try to only use the accelerometer or gyroscope data (not both), how does this
impact performance?


## Cloud Based ML: Edge Impulse




