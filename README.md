# Action-Recognition-TinyML-Edge_Impulse

 use machine learning to build a gesture recognition system that runs on a microcontroller. 

- The target gestures for classification are: 
  - **`StirringPot_LatchmanT`**
  - **`MakingTea_LatchmanT`**

## Gesture Recognition with Arduino Nano 33 BLE and TensorFlow Lite

Utilize an Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board to capture data from its onboard IMU (Inertial Measurement Unit) and classify movement as one of the predefined gestures using TensorFlow Lite for Microcontrollers.

## Gesture Recognition Training a TensorFlow Lite Micro Model For Arduino

Gesture Recognition Training a TensorFlow Lite Micro Model For Arduino where a custom model is trained
on data captured from an Arduino Nano 33 BLE Sense.

### i. Gesture Selection & Data Recording
### Streaming Sensor Data From the Arduino Board



You can seamlessly retrieve sensor data logs directly from the Arduino board. This can be achieved utilizing the same USB cable that's typically employed to program the board, connected to your laptop or PC.

- Two common kitchen tasks were used as Chosen gestures. selected actions attach the nano-sense BLE to your hand and record several instances of the actions

- Attach the nano-sense BLE to your hand and record several instances of these actions.

- Name your data files as `ActionName_YourLastNameFirstNamelnitial.csv`.

- Capturing Gesture Training Data
To capture data as a CSV log to upload to TensorFlow



### ii. Data Visualization
capturing the data by visualizing the data in the Arduino device. 
- Ensure your data is being captured correctly by visualizing it in the Arduino device.

Visualizing Live Sensor Data Log From the Arduino Board
With that done we can now visualize the data coming off the board.



### iii. Model Training in TensorFlow

Upload the punch.csv and flex.csv data
Parse and prepare the data
Build and train the model
Convert the trained model to TensorFlow Lite
Encode the model in an Arduino header file
The final step of the colab is generates the model.h file 

Train your model using TensorFlow.
Upload model to Arduino device and do inference on device displaying result on a
screen


### iv. Inference on Arduino Device

**`imu_classifier.ino`**
- Upload the trained model to your Arduino device.
- Perform inference directly on the device and display the results on a screen.

- Continually checks for notable motion through acceleration values extracted from the IMU.
- With sufficient samples, the TensorFlow Lite model gets activated to categorize the gesture.
- Subsequently, the outcome, representing the likelihoods of each gesture, is outputted to the Serial monitor.
  
- The TensorFlow Lite model in use can be located in the `model.h` file.
- It's plausible that this model underwent training outside using IMU data representing the gestures before being converted to the TensorFlow Lite format.
2. Incorporate the `model.h` file that contains the TensorFlow Lite model.
3. Upload the code onto an Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.
4. Manipulate the board matching the patterns of the recognized gestures and observe the classification outcomes on the Serial monitor.

  Perform some gestures
The confidence of each gesture will be printed to the Serial Monitor (0 = low confidence, 1 = high confidence)

### v. Data Quantity Experimentation

Experiment by increasing or decreasing the number of recordings per gesture. Analyze and note how this variation impacts the performance.


### vi. Data Source Experimentation

Experiment with using only the accelerometer or gyroscope data (not both). Analyze and determine the impact on performance.

increasing and decreasing the number of recordings per gesture, how does this
impact performance?

Try to only use the accelerometer or gyroscope data (not both), how does this
impact performance?


## Cloud Based ML: Edge Impulse


collect high-frequency data from real sensors, use signal processing to clean up data, build a neural network classifier, and how to deploy your model back to a device.

### Collecting your first data
With your device connected, we can collect some data
sampling move your device up and down in a continuous motion. 

### Design Impulse
An impulse takes the raw data, slices it up in smaller windows, uses signal processing blocks to extract features, and then uses a learning block to classify new data.

#### Spectral Analysis Block

'Spectral analysis' signal processing block. This block applies a filter, performs spectral analysis on the signal, and extracts frequency and spectral power data. Then we'll use a 'Neural Network' learning block, that takes these spectral features and learns to distinguish

Filter response - If you have chosen a filter (with non zero order), this will show you the response across frequencies. That is, it will show you how much each frequency will be attenuated.
After filter - the signal after applying the filter. This will remove noise.
Spectral power - the frequencies at which the signal is repeating 

#### Configuring the neural network
With all data processed it's time to start training a neural network. Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. The network that we're training here will take the signal processing data as an input, and try to map 
