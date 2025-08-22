🧠 CIFAR-10 Image Classification with CNN
A deep learning project using Convolutional Neural Networks (CNNs) to classify images from the CIFAR-10 dataset and predict custom uploaded images. Built with TensorFlow and designed for educational and portfolio use.

📦 Dataset
CIFAR-10: 60,000 32×32 color images in 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

Custom Uploads: You can upload your own images for prediction after training.

**🏗️ Model Architecture**
python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(11, activation='softmax')  # 10 CIFAR classes + 1 extra
])
⚠️ Note: The model includes an 11th class ("Human") but CIFAR-10 does not contain training data for it. For accurate predictions, consider training on a custom dataset with labeled human images.

🚀 How to Run
1. Install Dependencies
bash
pip install tensorflow pillow matplotlib
2. Train the Model
python
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
3. Save the Model
python
model.save("cifar10_model.h5")
4. Upload & Predict Custom Image
python
uploaded = files.upload()
# Automatically resizes, normalizes, and predicts uploaded image
📊 Output Example
Prediction: dog

Confidence: 0.91

Visualization: Bar chart showing confidence scores for all classes

🧪 Evaluation Metrics
Metric	Value (approx.)
Training Accuracy	~92%
Validation Accuracy	~88%
Loss	< 0.5
📌 To-Do / Ideas
[ ] Add data augmentation for better generalization

[ ] Replace CIFAR-10 with custom dataset including "Human" class

[ ] Deploy model using Streamlit or Hugging Face Spaces



[ ] Add bilingual (Roman Urdu + English) dashboard for accessibility

👩‍💻 Author
Javeria Iqbal Bachelor’s in Artificial Intelligence — Dawood University


**DEPLOY ON STREAMLIT**
<img width="964" height="644" alt="image" src="https://github.com/user-attachments/assets/0e67b514-9c3c-4a79-8053-2ba45bea8418" />
<img width="925" height="490" alt="image" src="https://github.com/user-attachments/assets/c4571511-7ca7-4120-aff4-fce42c65f456" />

