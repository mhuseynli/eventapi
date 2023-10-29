from django.db import models


class Webshop(models.Model):
    name = models.CharField(max_length=100)

    # Add any additional fields relevant to your webshop model

    def __str__(self):
        return self.name


class User(models.Model):
    webshop = models.ForeignKey(Webshop, on_delete=models.CASCADE)
    webshop_user_id = models.CharField(max_length=100)  # New field

    # Add any additional fields relevant to your user model

    def __str__(self):
        return self.name


class Product(models.Model):
    webshop = models.ForeignKey(Webshop, on_delete=models.CASCADE)
    webshop_product_id = models.CharField(max_length=100)
    tag = models.CharField(max_length=100)

    # Add any additional fields relevant to your product model

    def __str__(self):
        return f"{self.webshop.name} - {self.webshop_product_id}"


class Event(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    webshop = models.ForeignKey(Webshop, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)  # New foreign key
    event_type = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)

    # Add any additional fields relevant to your event model

    def __str__(self):
        return f"{self.user} - {self.event_type}"
