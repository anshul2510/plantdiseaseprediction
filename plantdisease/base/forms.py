from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm


class RegistrationForm(UserCreationForm):
    

    class Meta:
        model = User
        fields = (
                   'username',
                   'password1',
                   'password2'
               

        )

    def save(self,commit=True):
        user = super(RegistrationForm,self).save(commit=False)
        


        if commit:
            user.save()

        return user

""" class Meta:
        model = User
        fields = (
                   'username',
                   'password1',
                   'password2'
               

        )

    def save(self,commit=True):
        user = super(RegistrationForm,self).save(commit=False)
        self.fields['password1'].help_text = None


        if commit:
            user.save()

        return 
        
        """