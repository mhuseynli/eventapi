# serializers.py

from rest_framework import serializers
from .models import Webshop, Event


class WebshopSerializer(serializers.ModelSerializer):
    class Meta:
        model = Webshop
        fields = '__all__'


class EventSerializer(serializers.ModelSerializer):
    class Meta:
        model = Event
        fields = '__all__'
