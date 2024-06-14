from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageURLSerializer
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import io
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# Load and preprocess the dataset
(train_images, train_labels), (images, labels) = datasets.cifar10.load_data()
train_images, images = train_images / 255.0, images / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, 
          validation_data=(images, labels))

loss, accurate = model.evaluate(images, labels, verbose=2)

model.save('cifar10_model.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@api_view(['POST'])
def classify_image(request):
    serializer = ImageURLSerializer(data=request.data)
    if serializer.is_valid():
        image_url = serializer.validated_data['imageURL']
        
        response_image = requests.get(image_url)
        img = Image.open(io.BytesIO(response_image.content))
        img = img.resize((32, 32))  
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict the class
        predictions = model.predict(img)
        score = tf.nn.softmax(predictions[0])
        class_idx = np.argmax(score)
        confidence = score[class_idx].numpy()
        response_data =  {"Class": class_names[class_idx], "Accuracy": f"{confidence:.2f}"}
        return Response(
                status=status.HTTP_200_OK,
                data={
                    "message": "Created successfully",
                    "data": response_data
                }
            )

    else:
        return Response(
                status=status.HTTP_400_BAD_REQUEST,
                data={
                    "message": "Validation Error",
                    "data": serializer.errors
                }
            )
