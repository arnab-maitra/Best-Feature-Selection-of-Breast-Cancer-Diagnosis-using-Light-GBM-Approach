# Best Feature Selection of Breast Cancer Diagnosis using Light Gradient Boost (LGBoost) Approach

## Abstract

Breast cancer, a life-threatening disease affecting millions worldwide, poses significant challenges due to its time-consuming manual determination process, potential risks, and human errors. It is a condition where cells of the breast develop unnaturally and uncontrollably, resulting in a mass called a tumor. If lumps in the breast are not addressed,they can spread to other regions of the body, including the bones, liver, and lungs. Early diagnosis is crucial for effective treatment and improved patient outcomes. We focused on employing machine learning models to achieve quick identification of breast cancer tumors as benign or malignant. The primary objective is to develop a decision-making visualization pattern using swarm plots and heat maps. To accomplish this, we have utilized the Light GBM (Gradient Boosting Machine) algorithm and evaluated the model performance.

## Introduction

Today, Breast cancer is affecting individuals, particularly women. According to the World Health Organization (WHO). It's a leading cause of female mortality. Around a million women succumb to breast cancer annually with India's fatality rate at 13.92%. The prevalence is higher in Australia, Europe and the US, while Malaysia observes later-stage presentations. Regular screening is vital due to asymptomatic cases. Early detection aids treatment and survival. Contributing factors include family history, obesity, radiation exposure, and genetics. Recently discovered, breast cancer is categorized as malignant or benign. Analyzing tumor characteristics helps differentiate them. Benign tumors are low-risk, while malignant ones spread to neighboring tissues and the body. Artificial Intelligence (AI) is being employed to classify breast cancer. AI algorithms train on datasets to label tumors as 1 for benign or 0 for malignant.

## Motivation

The initial aim of this study is to examine breast cancer data derived from a diagnostic dataset comprising 30 feature columns and approximately 570 rows. The primary goal is to identify common characteristics in these groups that distinguish benign cases from malignant ones effectively. Subsequently, we plan to generate a heatmap visualization to identify and eliminate redundant features from the dataset. Finally, our ultimate objective is to create a machine learning model that enables users to classify breast cancer cases as either benign or malignant accurately. By accomplishing these objectives, we hope to enhance the diagnostic process and contribute to more efficient and precise breast cancer classification.

Our project aims to address challenges and propose solutions to enhance accuracy in breast cancer classification. Accuracy is a critical factor, as an imprecise model can lead to suboptimal outcomes. The report primarily centers around improving the accuracy of various algorithms, namely Logistic Regression, Gradient Boosting Algorithm, Random Forest Algorithm (Octaviani and Z Rustam et al.) [4], XG Boost Algorithm, and Light GBM Algorithm. The objective is to achieve the highest possible accuracy for the model by Light GBM algorithms. By tackling accuracy-related issues, we aspire to provide more reliable and effective breast cancer classification results.

## Project Description

Currently, India reports approximately 178,000 cases of breast cancer. However, manually determining cancer in these cases is an arduous and time-consuming process, often leading to delays and the possibility of human errors. To address this issue, we aim to develop a predictive model that can efficiently classify breast tumors as either malignant or benign using Machine Learning techniques. Our approach involves analyzing the correlation between various features, eliminating redundant data, and ultimately creating a highly accurate model. By leveraging these advanced technologies, we strive to enhance the early detection and diagnosis of breast cancer, which can significantly improve patient outcomes.

## Proposed Model / Approach

The approach proposed is Light Gradient Boosting (Light GBM Approach).
 
A novel approach in breast cancer detection has been introduced utilizing the Light Gradient Boost machine learning technique. This innovative method aims to transform initially weak learners into robust ones, thereby achieving enhanced accuracy in breast cancer detection. Unlike the conventional employment of weak learners as standalone classifiers, this technique leverages a boosting ensemble to achieve heightened classification accuracy. In this approach, the weak learners are harnessed as classifiers, which alone may not yield optimal classification accuracy. However, the concept of a strong learner emerges through the ensemble of these weak classifiers. This ensemble-based boosting technique is rooted in tree-based classification.

<p align="center">
  <img src="https://github.com/arnab-maitra/Best-Feature-Selection-of-Breast-Cancer-Diagnosis-using-Light-GBM-Approach/assets/88264132/b043d9ca-c763-4a6a-bf97-865059d6a99f" alt="Flowchart of LightGBM Algorithm" />
  <br><b>Flowchart of LightGBM Algorithm.</b>

Notably, the Light Gradient Boost machine learning technique molds the decision tree classifier into a unique weak learner structure, characterized by a vertical orientation. This innovative design, termed the "Leaf-wise Decision Tree Algorithm," showcases its distinctiveness in minimizing training loss compared to alternative algorithms. Through these advancements, the Light Gradient Boost technique demonstrates its potential to significantly improve breast cancer detection accuracy, thus offering promising avenues for enhanced medical diagnostics.

## Proposed Algorithm

https://carbon.now.sh/?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&width=680&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=4x&wm=false&code=%2523%2520Import%2520required%2520libraries%250A%2520import%2520numpy%2520as%2520np%250A%2520import%2520pandas%2520as%2520pd%250A%2520import%2520matplotlib.pyplot%2520as%2520plt%250A%2520from%2520matplotlib.pyplot%2520import%2520figure%250A%2520import%2520seaborn%2520as%2520sns%250A%2520import%2520lightgbm%2520as%2520lgb%250A%2520%2523%2520Load%2520the%2520data%250A%2520data%2520%253D%2520pd.read_csv%28%27data.csv%27%29%250A%2520%250A%2520data.head%28%29%250A%2520data.tail%28%29%250A%2520data.size%250A%2520data.shape%250A%2520%2523%2520Get%2520feature%2520names%250A%2520col%2520%253D%2520data.columns%250A%2520print%28col%29%250A%2520data.dtypes%250A%2520%2523%2520Target%2520variable%250A%2520y%2520%253D%2520data.diagnosis%2520%2523%2520M%2520or%2520B%250A%2520%2523%2520Features%250A%2520drop_list%2520%253D%2520%255B%27Unnamed%253A%252032%27%252C%2520%27id%27%252C%2520%27diagnosis%27%255D%250A%2520x%2520%253D%2520data.drop%28drop_list%252C%2520axis%253D1%29%250A%2520x.head%28%29%250A%2520%2523%2520Visualize%2520the%2520class%2520labels%250A%2520ax%2520%253D%2520sns.countplot%28x%253Dy%252C%2520label%253D%2522Count%2522%29%2520%2523%2520M%2520%253D%2520212%252C%2520B%2520%253D%2520357%250A%2520B%252C%2520M%2520%253D%2520y.value_counts%28%29%250A%2520print%28%27Number%2520of%2520Benign%253A%27%252C%2520B%29%250A%2520print%28%27Number%2520of%2520Malignant%253A%27%252C%2520M%29%250A%2520%2523%2520Correlation%2520map%250A%2520f%252C%2520ax%2520%253D%2520plt.subplots%28figsize%253D%2818%252C%252018%29%29%250A%2520sns.heatmap%28x.corr%28%29%252C%2520annot%253DTrue%252C%2520linewidths%253D.5%252C%2520fmt%253D%27.1f%27%252C%2520ax%253Dax%29%250A%2520drop_list1%2520%253D%2520%255B%27radius_mean%27%252C%2520%27concave%2520points_mean%27%252C%2520%27radius_se%27%252C%250A%2520%27texture_se%27%252C%250A%2520%27perimeter_se%27%252C%250A%2520%27compactness_se%27%252C%250A%2520%27area_se%27%252C%250A%2520%27concavity_se%27%252C%250A%2520%27symmetry_se%27%252C%250A%2520%27concave%250A%2520%27fractal_dimension_se%27%252C%250A%2520%27texture_worst%27%252C%250A%2520%27smoothness_worst%27%252C%250A%2520%27concave%250A%2520%27fractal_dimension_worst%27%255D%250A%2520%27perimeter_worst%27%252C%250A%2520%27compactness_worst%27%252C%250A%2520points_worst%27%252C%250A%2520x_1%2520%253D%2520x.drop%28drop_list1%252C%2520axis%253D1%29%250A%2520%2523%2520Correlation%2520heatmap%250A%2520f%252C%2520ax%2520%253D%2520plt.subplots%28figsize%253D%2812%252C%252012%29%29%250A%2520sns.heatmap%28x_1.corr%28%29%252C%250A%2520annot%253DTrue%252C%250A%2520%27smoothness_se%27%252C%250A%2520points_se%27%252C%250A%2520%27radius_worst%27%252C%250A%2520%27area_worst%27%252C%250A%2520%27concavity_worst%27%252C%250A%2520%27symmetry_worst%27%252C%250A%2520linewidths%253D.5%252C%250A%2520ax%253Dax%29%250A%2520from%2520sklearn.model_selection%2520import%2520train_test_split%250A%2520%2520%250A%2520%2520%2523Split%2520the%2520data%250A%2520x_train%252C%2520x_test%252C%2520y_train%252C%2520y_test%2520%253D%2520train_test_split%28x_1%252C%2520y%252C%250A%2520test_size%253D0.3%252C%2520random_state%253D42%29%250A%2520%2523%2520Initialize%2520the%2520model%250A%2520model%2520%253D%2520lgb.LGBMClassifier%28%29%250A%2520%2523%2520Train%2520the%2520model%250A%2520model.fit%28x_train%252C%2520y_train%29%250A%2520%2523%2520Predictions%2520on%2520the%2520test%2520set%250A%2520y_pred%2520%253D%2520model.predict%28x_test%29%250A%2520from%2520sklearn.metrics%2520import%2520accuracy_score%250A%2520%2523%2520Calculate%2520accuracy%250A%2520accuracy%2520%253D%2520accuracy_score%28y_test%252C%2520y_pred%29%250A%2520print%28%2522Accuracy%253A%2522%252C%2520accuracy%29

## Dataset


## Methodology


## Results


## Conclusion


## Future Work



![Screenshot 2024-02-22 001758](https://github.com/arnab-maitra/Keylogger/assets/88264132/1cca1f6d-d64b-4fe9-9b99-736107506358)
<b><p align="center">Entered sample text in the terminal while the keylogger is active.</p></b>

<br></br>

![Screenshot 2024-02-22 001820](https://github.com/arnab-maitra/Keylogger/assets/88264132/807205c1-a643-4f3e-bad0-888a4643f2a8)
![Screenshot 2024-02-22 001843](https://github.com/arnab-maitra/Keylogger/assets/88264132/a179ea84-f5e6-4fd7-831f-57cc4209a13f)
![Screenshot 2024-02-22 001900](https://github.com/arnab-maitra/Keylogger/assets/88264132/4c009ea2-d950-499d-8bbf-b3335fc35f78)
<b><p align="center">Keystrokes recorded by the keylogger program and captured input stored in the log file (keylog.txt).</p></b>
