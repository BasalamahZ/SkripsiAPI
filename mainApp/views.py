from django.utils import timezone
import requests
import cloudinary
import numpy as np
import io
import librosa
import tensorflow as tf
import soundfile as sf
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser 
from rest_framework import status
from .models import Users, Data
from .serializers import UsersSerializer, DataSerializer
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from sklearn.preprocessing import LabelEncoder
from mainApp import mfcc

def hello_world(request):
    return HttpResponse("Hello, World!")

# This function will return the users by id
@csrf_exempt
@api_view(['GET', 'POST'])
def user_list(request):
    if request.method == 'GET':
        users = Users.objects.all()
        email = request.GET.get('email', None)
        if email is not None:
            users = users.filter(email__icontains=email)
        users_serializer = UsersSerializer(users, many=True)
        return JsonResponse(users_serializer.data, safe=False)
        # 'safe=False' for objects serialization
 
    elif request.method == 'POST':
        user_data = JSONParser().parse(request)
        email = user_data['email']
        existing_user = Users.objects.all().filter(email=email).first()
        if existing_user:
                return JsonResponse(UsersSerializer(existing_user).data, status=status.HTTP_200_OK)
        user_serializer = UsersSerializer(data=user_data)
        if user_serializer.is_valid():
            user_serializer.save()
            return JsonResponse(user_serializer.data, status=status.HTTP_201_CREATED) 
        return JsonResponse(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def user_detail(request, id):
    try: 
        user = Users.objects.get(id=id) 
    except Users.DoesNotExist: 
        return JsonResponse({'message': 'user does not exist'}, status=status.HTTP_404_NOT_FOUND)
 
    if request.method == 'GET': 
        user_serializer = UsersSerializer(user) 
        return JsonResponse(user_serializer.data) 
    
# This function will return the users by id
@csrf_exempt
@api_view(['GET', 'POST'])
def data_list(request):
    if request.method == 'GET':
        data = Data.objects.all()
        data_serializer = DataSerializer(data, many=True)
        return JsonResponse(data_serializer.data, safe=False)
        # 'safe=False' for objects serialization
 
    elif request.method == 'POST':
        data = JSONParser().parse(request)
        data['date_created'] = timezone.now()
        sound_uri = data["sound_uri"]
        response = requests.get(sound_uri)
        if response.status_code != 200:
            return JsonResponse({'message':'error get sound uri'}, status=status.HTTP_400_BAD_REQUEST)
        
        y, sr = librosa.load(io.BytesIO(response.content), sr=24000)

        # Resample to 8000 Hz
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=8000)
        output_path = "./file_resampled.wav"
        sf.write(output_path, y_resampled, 8000)
        
        tf.keras.backend.clear_session()
        # load model
        if data['algorithm'] == 'galih':
            model = tf.keras.models.load_model("./model/model_spec_galih.h5", compile=False)
            predict_data = mfcc.prepare_data2([output_path])
        elif data['algorithm'] == 'zidane':
            model = tf.keras.models.load_model("./model/model_wavelet_zidane.h5", compile=False)
            predict_data = mfcc.prepare_data3([output_path])
        elif data['algorithm'] == 'izza':
            model = tf.keras.models.load_model("./model/model_mfcc_galih.h5", compile=False)
            predict_data = mfcc.prepare_data([output_path])
        elif data['algorithm'] == 'delfi':
            model = tf.keras.models.load_model("./model/model_mfcc_galih.h5", compile=False)
            predict_data = mfcc.prepare_data([output_path])
        else:
            return JsonResponse({'message':'algorithm not found'}, status=status.HTTP_400_BAD_REQUEST)

        # Predict and Data Classes
        predict_data = model.predict(predict_data, use_multiprocessing=True, workers=6, verbose=1)
        
        # Fit the encoder on your class labels
        class_labels = ["HighStress", "LowStress"]
        encoder = LabelEncoder()
        encoder.fit(class_labels)
        predicted_labels = encoder.inverse_transform([np.argmax(predict_data)])
        
        data['label'] = predicted_labels[0]
        data_serializer = DataSerializer(data=data)
        if data_serializer.is_valid():
            data_serializer.save()
            return JsonResponse(data_serializer.data, status=status.HTTP_201_CREATED) 
        return JsonResponse(data_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
@csrf_exempt
@api_view(['POST'])
def uploadSound(request):
    if request.method == 'POST':
            sound_upload = request.FILES['sound']
            try:
                sound = cloudinary.uploader.upload(sound_upload, resource_type="auto", folder="sound")
                return JsonResponse({'data': sound['secure_url']}, status=status.HTTP_201_CREATED) 
            except:
                return JsonResponse({'message': 'error'}, status=status.HTTP_400_BAD_REQUEST)
