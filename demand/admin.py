from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User

# Register the custom user model with the admin site
admin.site.register(User, UserAdmin)

