James Streett 

March 23, 2025

# Musically Trained CNNs for Genre Classification and

# Content-Based Song Recommendation

Abstract 

Music genre classification and recommendation systems has gained significant traction in recent years, particularly with the rise of machine learning techniques. This paper presents an approach using convolutional neural networks (CNNs) to classify songs into ten different genres based on audio features extracted from a Kaggle dataset. The dataset consists of 30-second clips with extracted numerical features using Librosa, including 19 general and 40 MFCC-related features. A carefully balanced 70/30 train-test split is implemented to ensure equal genre representation. After extensive experimentation with CNN architectures, the final model comprises three convolutional layers with batch normalization, dropout, and an early stopping mechanism to optimize performance while mitigating overfitting. The classification model achieves an accuracy of approximately 73%. Finally, a recommendation system is built using cosine similarity to suggest similar songs based on extracted feature embeddings. 

## 1. Introduction

With the exponential growth of digital music platforms, automated genre classification and recommendation have become essential for enhancing user experience. Traditional genre classification methods rely on handcrafted features and manual annotations, which are both time-consuming and error-prone. Deep learning techniques, particularly convolutional neural networks (CNNs), have demonstrated strong performance in pattern recognition tasks, including image and audio classification. This paper explores the effectiveness of a CNN-based model for music genre classification using a Kaggle dataset consisting of ten genres, followed by a content-based recommendation system leveraging cosine similarity. 

## 2. Methods and Model Architecture 

### 2.1 Dataset and Preprocessing

The dataset contains 100 samples per genre, totaling 1,000 audio clips, each 30 seconds long. Librosa is used to extract numerical features, resulting in a dataset with 60 feature columns: 19 general features and 40 MFCC-related features. These features are stored in a CSV file. To ensure an evenly distributed train-test split, a stratified sampling approach is used to allocate 70% of each genre to training and 30% to testing. 

#### CSV File Includes:

filename,length,chroma_stft_mean,chroma_stft_var,rms_mean,rms_var,spectral_centroid_mean,spectral_centroid_var,spectral_bandwidth_mean,spectral_bandwidth_var,rolloff_mean,rolloff_va r,zero_crossing_rate_mean,zero_crossing_rate_var,harmony_mean,harmony_var,perceptr_mean,perceptr_var,tempo,mfcc1_mean,mfcc1_var,mfcc2_mean,mfcc2_var,mfcc3_mean,mfcc3_var,mf cc4_mean,mfcc4_var,mfcc5_mean,mfcc5_var,mfcc6_mean,mfcc6_var,mfcc7_mean,mfcc7_var,mfcc8_mean,mfcc8_var,mfcc9_mean,mfcc9_var,mfcc10_mean,mfcc10_var,mfcc11_mean,mfcc 11_var,mfcc12_mean,mfcc12_var,mfcc13_mean,mfcc13_var,mfcc14_mean,mfcc14_var,mfcc15_mean,mfcc15_var,mfcc16_mean,mfcc16_var,mfcc17_mean,mfcc17_var,mfcc18_mean,mfcc18_var,mfcc19_mean,mfcc19_var,mfcc20_mean,mfcc20_var,label 

A critical flaw in previous approaches done on Kaggle was that when splitting the dataset, the 30-second clips were divided into 10 smaller 3-second clips. This led to instances of the same song appearing in both training and testing sets, causing data leakage and overfitting. In this study, careful preprocessing ensures that entire 30-second clips remain within a single set, preserving data integrity. 

Feature selection was crucial in ensuring the model learns meaningful representations. MFCCs (Mel-frequency cepstral coefficients) were chosen because they effectively capture timbral characteristics of music, which are essential for distinguishing genres. Additionally, spectral contrast, zero-crossing rate, and chroma features were included to enhance the model’s ability to differentiate similar genres. 

A spectrogram is a visual representation of the spectrum of frequencies in a signal as it varies with time. A mel spectrogram is a variant where the frequency axis is mapped to the mel scale, which better represents human auditory perception. Mel spectrograms help capture genre-specific characteristics more effectively than raw waveforms. ​​


![](https://web-api.textin.com/ocr_image/external/5468c4e7f46787b7.jpg)

Figure 1: Rock Mel Spectrogram: rock.00016.wav 

Before building the model, it is essential to explore the dataset to understand feature relationships. One way to do this is by generating a correlation heatmap of the mean values of extracted features. This helps identify potential redundancies and dependencies between features,allowing for better feature selection and dimensionality reduction. 


![](https://web-api.textin.com/ocr_image/external/3ac7a8ff1c97e757.jpg)

Figure 2: Mean Feature Correlation HeatMap

The heatmap reveals strong correlations between some MFCC-related features, indicating that certain features may contribute similar information to the model. For example MFCC2 and rms_mean exhibit high correlation, which suggests they capture similar aspects of the audio signal. Understanding these relationships helps optimize feature selection and avoid overfitting.

2.2 Convolutional Neural Network (CNN) Model with Tensorflow

Layer 1: 64 filters, kernel size = 3, ReLU activation, followed by batch normalization.

Layer 2: 128 filters, kernel size = 3, batch normalization followed by ReLU activation.

Layer 3: 128 filters, kernel s\text{ize}=2, ReLU activation, followed by max pooling (pool size = 2).

Flatten layer: Converts the feature maps into a dense vector.

Dense Layer: 128 neurons with ReLU activation.

Dropout: Set to 0.6 to reduce overfitting.

Output Layer: Softmax activation with 10 outputs (one for each genre).

The choice of kernel sizes (3, 3, and 2) was based on performance tuning, balancing accuracy and overfitting concerns. Batch normalization was applied to stabilize training and speed up convergence. BN before ReLU (in the second Conv layer) helped stabilize training and prevent neurons from dying. Instead of BN, we use Max Pooling. The model has already normalized activations from the previous layers. Max pooling was introduced in the third layer to reduce spatial dimensions and prevent overfitting. 

### 2.3 Training and Optimization

The model was trained using Adam with a learning rate of 0.002. Several optimizers, including stochastic gradient descent (SGD) were tested, but Adam provided better generalization for CNN-based classification tasks. Early stopping was implemented to monitor validation accuracy and prevent overfitting, restoring the best model after 20 epochs of no improvement. 

## 3. Experiments and Results 

### 3.1 Training Process and Overfitting Management 

Originally, with two convolutional layers, validation accuracy remained significantly lower than training accuracy, indicating overfitting. Then with four convolutional layers, the training accuracy began to skyrocket, clearly overfitting with greater than 98% accuracy. Thus giving support towards 3 convolutional layers. By adding batch normalization after the second convolutional layer with activation applied afterward, the network yielded better results than applying activation before normalization. The best configuration resulted in a test accuracy of approximately 73%. 

At the end of 20 epochs, the model showed signs of mild overfitting, with training accuracy reaching 80% while validation accuracy plateaued at 73%. This suggests that while the model generalizes well, additional techniques such as data augmentation or regularization could further mitigate some minor overfitting. 

3.2 Evaluation Metrics

To further analyze the model's performance, a confusion matrix was generated, showing the distribution of correct and misclassified genres. The confusion matrix highlights which genres are more frequently confused, providing insight into potential areas for improvement. 


![](https://web-api.textin.com/ocr_image/external/d868c7a427600753.jpg)

Figure 3: Confusion Matrix

With the majority of the classifications being correct, areas such as the intersection of blues and country shows that the model incorrectly classified 7 instances of blues as country. However most correct classifications are well suited with a low of 18 and a high of 29. 

Additionally, training accuracy and loss graphs were plotted to visualize the learning process over 20 epochs, of course with the use of early stopping. 


![](https://web-api.textin.com/ocr_image/external/6b3377973e5000c7.jpg)

Figure 4: Training and Validation Accuracy 


![](https://web-api.textin.com/ocr_image/external/9c986a66095d559d.jpg)

Figure 5: Training and Validation Loss

In Figure 4 and Figure 5 we can see a small gap between the Training Accuracy and Validation Accuracy, suggesting a bit of overfitting. With some more regularization, the data could probably be better fitted but as for a 3 layer Convolutional Network this is the closest tested solution. 

### 3.3 Content-Based Music Recommendation Using Cosine Similarity

In addition to genre classification, this study implements a content-based recommendation system using cosine similarity. The goal is to identify songs with similar audio characteristics by comparing feature embeddings extracted from the trained CNN model. 

Once the CNN model is trained, an intermediate feature representation is extracted from the flatten layer of the network. This representation serves as a high-dimensional embedding that captures important audio characteristics for each song. Using the trained model, embeddings are generated for the entire dataset, effectively transforming each song into a numerical vector suitable for similarity calculations. 

The recommendation system randomly selects a song and computes cosine similarity between its feature embedding and all other songs in the dataset. The top five most similar songs are retrieved based on similarity scores. The cosine similarity between two feature vectors A and B is calculated as follows: 

cosine similari\mathrm{ty}=\mathrm{A}·B/||A||·||B||

This metric quantifies how close two songs are in the feature space, with values ranging from -1(completely dissimilar) to 1 (identical). 

The implementation follows these steps:

1.​ Extract feature embeddings from the CNN's flatten layer.

2.​ Compute cosine similarity between a randomly selected song and all other songs

## 3.​ Retrieve and display the five most similar songs.

Figure 6: Song Recommendations

This approach effectively groups songs with similar audio characteristics. As we can tell jazz 98, is most similar to jazz 11, with a 0.7826 accuracy. The retrieved recommendations tend to belong to the same genre, reinforcing CNN's ability to learn meaningful feature representations. A key advantage of this method is its ability to generalize to new songs without requiring genre labels—useful for personalized recommendations on streaming platforms. 

## 5. Bonus: Novelty and Contributions 

Unlike previous studies that incorrectly split 30-second clips into 3-second fragments across train-test sets, this work ensures proper data integrity, preventing information leakage. Various architectures, optimizers, and hyperparameters were tested to achieve optimal performance. Implementing dropout, batch normalization, and early stopping contributed to better generalization. Traditional models like SVM and XGBoost were tested but showed lower performance, validating the effectiveness of CNNs. 

## 6. Conclusion and Future Work

With this paper, it shows the potential of CNNs for music genre classification and recommendation. Through careful tuning of convolutional layers, kernel sizes, dropout rates, and normalization techniques, the model achieves strong performance while mitigating overfitting. The use of a balanced train-test split ensures robustness across all genres. Additionally, the recommendation system using cosine similarity provides a meaningful way to suggest similar songs based on extracted embeddings. 

In the future, this study can expand on and include data augmentation, transformer-based architectures, hybrid recommendation systems, and feature expansion to enhance model accuracy and less overfitting. 

## 7. References 

●​ https://www.kaggle.com/code/andradaolteanu/work-w-audio-data-visualise-classify-reco mmend/notebook#EDA 

●​ https://machinelearningmastery.com/improve-deep-learning-performance/ 

●​ UCSD Cogs 181 Teachings, Professor Zhuowen Tu 



