from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from kobert.load_model import predict
from django.urls import reverse
import json

def index(request):
    return redirect('/emotion')
    #return render(request, 'emotion.html')
    #return HttpResponseRedirect(reverse('BERT:emotion'))

def emotion(request):
    if request.method=='POST':
        #text = request.POST.get("text")
        data = json.loads(request.body)
        text = data["text"]
        emotion = predict(text)
        print(emotion)
        context = {"emotion":emotion}
        return  JsonResponse(context)
    else:
        return render(request, 'emotion.html')
    

def emotion_service(request, text:str):
    emotion = predict(text)
    #enc_user_emotion = emotion.decode('utf-8').encode('cp949')
    context = {"emotion":emotion}
    print(emotion)
    return  JsonResponse(context)

