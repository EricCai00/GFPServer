from django.shortcuts import render
from django.http import HttpResponse
from django.core.files import File
import logging
import subprocess
import random
import os
import time

logger = logging.getLogger('django')


def save_dir():
    local_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    output_path = os.path.join('C:\\Workspace\\PE\\UI\\files\\', local_time)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        output_path = output_path + '-' + str(random.randint(1, 1000))
        os.makedirs(output_path)
    return output_path


# Create your views here.
def home(request):
    return render(request, 'home.html')


def download(request, filename):
    file_pathname = os.path.join(output_path, filename)

    with open(file_pathname, 'rb') as f:
        file = File(f)
        response = HttpResponse(file.chunks(), content_type='APPLICATION/OCTET-STREAM')
        response['Content-Disposition'] = 'attachment; filename=' + filename
        response['Content-Length'] = os.path.getsize(file_pathname)
    # os.unlink(file_pathname)
    return response


def calculate(request):
    global output_path
    output_path = save_dir()
    files = request.FILES.getlist('filename')
    if not files:
        return render(request, 'home.html')

    for file in files:
        input_file = os.path.join(output_path, file.name)
        destination = open(input_file, 'wb+')
        for chunk in file.chunks():
            destination.write(chunk)

        destination.close()

    shell = 'python C:\\Workspace\\PE\\UI\\predict\\main.py ' + input_file
    subprocess.run(shell, shell=True)       # stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    # print('STDOUT:', stdout, 'STDERR:', stderr)
    # while not stdout.endswith(b'Completed\r\n'):
    #     print("WAITING")
    #     time.sleep(0.5)

    file = 'results.txt'

    return render(request, 'download.html', {'file': file, 'message': 123})


def test(request):
    message = request.GET
    print('MESSAGE:', message)
    return render(request, 'test.html', {'message': message})
