import torch
from torch import nn
from torchvision import  transforms, models
from PIL import Image
import sqlscript
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = models.alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_ftrs=model.classifier[6].in_features
model.classifier[6]= nn.Linear(num_ftrs, 6)
fc_parameters = model.classifier[6].parameters()
for param in fc_parameters:
    param.requires_grad = True
use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()
class_names = ['abdulaziz', 'Akshay Kumar', 'Alia Bhatt', 'Natalie Portman', 'Roger Federer', 'Vijay Deverakonda']
model.load_state_dict(torch.load('alexnet.pt'), strict=False)

prediction_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(192),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
# discard the transparent, alpha channel (that's the :3) and add the batch dimension
model = model.cpu()
model.eval()
#imagecs=cv2.imread(img_pth)
font = cv2.FONT_HERSHEY_SIMPLEX
# org
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2
vid = cv2.VideoCapture(0)
conn=sqlscript.database_conn()
while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    try:
        for (x, y, w, h) in faces:
            cv2.imwrite('pred.png', frame[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image = Image.open('pred.png')
            image = prediction_transform(image)[:3, :, :].unsqueeze(0)
            # Display the resulting frame
            out=model(image)
            idx = torch.argmax(out)  ### print the result of model(image) then print the result of argmax.torch method
            _, index = torch.max(out, 1)
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
            per=percentage[index[0]].item()
            print(class_names[index[0]], percentage[index[0]].item())
            print(class_names[idx])
            if( per>70):
                # Using cv2.putText() method
                image = cv2.putText(frame, class_names[idx],(x, y) , font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                emp_no=1+int(idx.item())
                sqlscript.insert_emp(conn,emp_no,1)
    except:
        pass
    cv2.imshow('frame', frame)
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break