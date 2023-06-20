from django.shortcuts import redirect
from django.urls import reverse

class RestrictLoggedInMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.user.is_authenticated:
            if request.path == reverse('home'):
                return redirect(reverse('login'))
            

        else:
            if request.path == reverse('register') or request.path == reverse('login'):
                return redirect(reverse('home'))

        response = self.get_response(request)
        return response