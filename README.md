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

- *Stirring Pot Visualization*
![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/83b99636-6e17-45ec-9bf3-e63243ead9bf)


- *Making Tea Visualization*
![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/1520ab1d-a423-493e-b5a7-9169b61c3bc4)






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

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/3f760058-f7dd-4e43-8857-8e476798d0a7)



### v. Data Quantity Experimentation


#### Experiment by increasing or decreasing the number of recordings per gesture. Analyze and note how this variation impacts the performance.

- Originally we have const int SAMPLES_PER_GESTURE = 119; which has the following number of recordings:
![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/ad23e851-68df-4be1-8351-01d92590948a)


And had the following performance in training:
![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/930e089a-db30-4673-969c-57bc6ef981f4)


- I experimented with 200 for increasing samples, and 50 for decreasing samples.

- When making Samples per gesture = 200, the model was less accurate and didn’t perform well, and there were less recordings to learn from. 

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/04b3dda0-6af2-4fe0-a191-030a51a5b935)


- In training the model

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/909c3b4e-fc81-4fef-8716-9ef8e0ff4db1)


- For number of samples per gesture = 50, there were more samples to learn from
  
![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/910f1f35-379b-4d64-831a-4c3c3cce1312)


- decreasing number of recordings per gesture made for a better/comparable model!

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/7c813676-3135-472b-844a-b4fbe2e51ba9)




### vi. Data Source Experimentation

#### Experiment with using only the accelerometer or gyroscope data (not both). Analyze and determine the impact on performance.

- *When using strictly gyroscopic, the model was not able to recognize gestures well.*

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/032d61a7-8ebf-4746-923f-dbd71df37efd)

- When using strictly accelerometer data, the model wasn’t perfect, but was more accurate than strictly using gyroscopic data. *

  ![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/9dee0a96-0e53-46ab-88c5-7d13524cc8ea)





## Cloud Based ML: Edge Impulse

Perform the kitchen actions that you did in Lab 2 to collect data using EdgeImpulse, build a model and run inference on Arduino Nano 33 BLE Sense.

collect high-frequency data from real sensors, use signal processing to clean up data, build a neural network classifier, and how to deploy your model back to a device.

### Collecting your first data
With your device connected, we can collect some data
sampling move your device up and down in a continuous motion. 

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/8b3ec8a4-7add-4e22-9f78-1de8eb801f52)

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/59514dd9-c047-4bee-8ee2-a002963d52db)


### Design Impulse
An impulse takes the raw data, slices it up in smaller windows, uses signal processing blocks to extract features, and then uses a learning block to classify new data.

#### Pre-processing for Action Recognition Models - Data Filtering

'Spectral analysis' signal processing block that applies a filter, performs spectral analysis on the signal, and extracts frequency and spectral power data. Raw data is frequently characterized by noise, which can be detrimental to action recognition models. Applying data smoothing filters to raw input data can greatly enhance model performance.

I included the folders that allowed me to deploy custom preprocessing blocks to the Edge Web Portal. For each folder, the main files that were changed were  **`parameters.json`** and **`dsp.py`**

- *Finite Impulse Response (FIR) filter*

 See **`parameters.json`** and **`dsp.py`** within the **`FIR`** folder for custom block design.

 ![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/9ab4f90e-ebeb-4da9-8cbb-045c9d93c6c7)

  
- *Kalman filter*
See **`parameters.json`** and **`dsp.py`** within the **`Kalman 1`** folder and  **`Kalman 2`** folder for implementation of custom block designs. recursive Bayesian filter 

**`Kalman 2`** initializes all System State variables, and then predicting and updating). **`Kalman 1`**  just uses an imported package. 

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/545afa38-22f0-45f8-a75b-906922a10d2b)

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/ffdab362-4595-430d-b182-3470fce2fb43)



#### Configuring the neural network
With all data processed it's time to start training a neural network. Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. The network that we're training here will take the signal processing data as an input, and try to map 

- *Training Model (Default NN Architecture)*
![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/fea09eea-f08e-4c04-866c-8ade0d9ebac6)  


![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/54ceb5b3-646e-4100-8cd3-8cb9cbf8ef2d)  


![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/f7621a53-19ae-47c1-bef8-c045ccd7a499)  


![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/6a1a72b3-e875-4181-9389-487172ea4be0)  
  

### Embedded AI/ML with EdgeImpulse

Most AI/ML applications on embedded devices rely heavily on pre-processing steps that are necessary before using neural networks for classification. This pre-processing is often referred to as **Digital Signal Processing (DSP)**.

Using EdgeImpulse's built-in EON compiler allows for architecture exploration for both the DSP and the NN (Neural Network) inference subsystems. For the action recognition task, there are various trade-offs to consider:

## Key Exploration Areas

- **Neural Network Model Complexity**: Explore three different neural network models.
  
- **Quantization and Framework Comparison**: Compare using quantized (INT8) versus floating-point (FP) representations. Specifically, contrast TensorFlowLite Micro (TFLM) with EdgeImpulse's EON. When comparing, showcase:
  - Resources Used: RAM and Flash
  - Model Accuracy


| Model | Representation | Framework | Accuracy | RAM (KB) | Flash (KB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Model 1 | INT8 | TFLM | $99.42 \%$ | $2.7 \mathrm{~K}$ | $33.7 \mathrm{~K}$ |
| Model 1 | INT8 | EON | $99.42 \%$ | $1.8 \mathrm{~K}$ | $15.4 \mathrm{~K}$ |
| Model 1 | FP | TFLM | $99.42 \%$ | $2.8 \mathrm{~K}$ | $36.1 \mathrm{~K}$ |
| Model 1 | FP | EON | $99.42 \%$ | $1.8 \mathrm{~K}$ | $14.1 \mathrm{~K}$ |
| Model 2 | INT8 | TFLM | $99.42 \%$ | $4.3 \mathrm{~K}$ | $51.5 \mathrm{~K}$ |
| Model 2 | INT8 | EON | $99.42 \%$ | $3.4 \mathrm{~K}$ | $28.6 \mathrm{~K}$ |
| Model 2 | FP | TFLM | $99.42 \%$ | $5.6 \mathrm{~K}$ | $57.7 \mathrm{~K}$ |
| Model 2 | FP | EON | $99.42 \%$ | $4.3 \mathrm{~K}$ | $26.6 \mathrm{~K}$ |
| Model 3 | INT8 | TFLM | $99.34 \%$ | $6.7 \mathrm{~K}$ | $83.6 \mathrm{~K}$ |
| Model 3 | INT8 | EON | $99.34 \%$ | $7.3 \mathrm{~K}$ | $59.8 \mathrm{~K}$ |
| Model 3 | FP | TFLM | $99.26 \%$ | $11.6 \mathrm{~K}$ | $176.0 \mathrm{~K}$ |
| Model 3 | FP | EON | $99.26 \%$ | $9.6 \mathrm{~K}$ | $144.5 \mathrm{~K}$ |

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/999b33b0-1ec3-4dda-b83b-1c0df3f97ebf)  

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/08d54ef7-6ba0-4f8f-90f7-27a6a09117e1)

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/b31fdb1a-4b0a-437a-a965-66b89bbbf23e)

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/e072b91b-b6fe-4a79-88a2-eb989f04713e)

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/b1ce2aab-5cb9-4bef-b8a8-0e33f91898c9)

![image](https://github.com/travislatchman/Action-Recognition-TinyML-Edge_Impulse/assets/32372013/21cfd4dc-cc2a-44f4-8a62-5a2bd0935f75)


1. **Accuracy**:
    - Model 1 and Model 2 both achieve an accuracy of 99.42%.
    - Model 3's accuracy is slightly lower, at 99.34% for INT8 representation and 99.26% for FP representation.
    - Notably, there is a negligible difference in accuracy between the INT8 and FP representations for each model.

2. **RAM and Flash Usage**:
    - In general, as model complexity and size increase, both RAM and Flash usage also increase. In other words, the usage is in the order: Model 1 < Model 2 < Model 3.

3. **Numerical Representation**:
    - The INT8 representation consistently exhibits lower RAM and Flash usage compared to the FP representation when evaluated on the same model and framework.
    - This is anticipated since INT8 requires fewer memory resources to store numbers compared to floating-point representations.

4. **Framework Comparison**:
    - Across all models and numerical representations, the EON framework demonstrates lower RAM and Flash memory consumption than the TFLM framework.
    - This suggests that EON might offer a more memory-efficient alternative for model deployment.


