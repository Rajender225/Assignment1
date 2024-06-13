from rest_framework import serializers

class ImageURLSerializer(serializers.Serializer):
    imageURL = serializers.URLField()