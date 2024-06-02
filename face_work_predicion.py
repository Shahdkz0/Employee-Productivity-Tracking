import torch
from torch import nn
from torchvision import  transforms, models
from PIL import Image
import sqlscript
import cv2
import time
from datetime import date,datetime
today_time=datetime.now()
current_time = today_time.strftime('%H:%M:%S')
time_start = datetime.strptime('08:00:00', '%H:%M:%S')
time_arrive = datetime.strptime('08:30:00', '%H:%M:%S')
time_leave = datetime.strptime('14:00:00', '%H:%M:%S')
time_start=time_start.strftime('%H:%M:%S')
time_arrive=time_arrive.strftime('%H:%M:%S')
time_leave=time_leave.strftime('%H:%M:%S')
print(time_leave)
print(time_arrive)
def detect_and_draw(model, frame, start_time):
    # Convert BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(frame_rgb)

    # Get detected objects
    detections = results.xyxy[0]

    # Filter detections to keep only 'person' class
    person_detections = [detection for detection in detections if detection[5] == 0]  # Assuming 'person' is class 0

    # Current time
    current_time = time.time()

    # Calculate duration in seconds
    duration_seconds = current_time - start_time

    # Convert duration to hours, minutes, and seconds
    duration_hours = int(duration_seconds // 3600)
    duration_minutes = int((duration_seconds % 3600) // 60)
    duration_seconds = int(duration_seconds % 60)

    # Iterate over each detected person
    for detection in person_detections:
        # Extract coordinates
        x1, y1, x2, y2 = detection[:4].cpu().numpy().astype(int)  # Convert to numpy array and then to int

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put class name and time duration inside the bounding box
        class_name = "work"
        duration_text = f'{duration_hours:02}:{duration_minutes:02}:{duration_seconds:02}'

        # Calculate text sizes for positioning
        class_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        duration_size = cv2.getTextSize(duration_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

        # Position class name and time duration
        class_x = x1
        duration_x = x2 - duration_size[0]
        y = y1 - 10  # Y-coordinate for both texts

        # Put class name
        cv2.putText(frame, class_name, (class_x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Put time duration
        cv2.putText(frame, duration_text, (duration_x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, results  # Return both the processed frame and the results

def process_video(input_video, output_video):
    # Load YOLOv5 model
    yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Adjust confidence threshold
    conf_threshold = 0.2
    yolomodel.conf = conf_threshold

    # Adjust IoU threshold
    iou_threshold = 0.80
    yolomodel.iou = iou_threshold

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
class_names =['Vijay Deverakonda ', 'Akshay Kumar', 'Alia Bhatt', 'Natalie Portman', 'Roger Federer', 'abdulaziz']
model.load_state_dict(torch.load('alexnet.pt'), strict=False)
yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                       transforms.ToTensor()])
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
fps = vid.get(cv2.CAP_PROP_FPS)
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize start time
start_time = time.time()

# Define codec and VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

# Count of detected persons
person_count = 0
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
            out = model(image)
            idx = torch.argmax(out)  ### print the result of model(image) then print the result of argmax.torch method
            _, index = torch.max(out, 1)
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
            per = percentage[index[0]].item()
            print(class_names[index[0]], percentage[index[0]].item())
            print(class_names[idx])
            if (per > 70):
                # Using cv2.putText() method
                image = cv2.putText(frame, class_names[idx], (x, y), font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                emp_no = 1 + int(idx.item())
                if (current_time>=time_start and current_time<=time_arrive):
                        sqlscript.insert_emp(conn, emp_no, 1)
                elif (current_time>=time_leave):
                    sqlscript.insert_leav_emp(conn, emp_no, 2)

                elif (current_time>time_arrive and current_time<time_leave):
                    sqlscript.insert_emp(conn, emp_no, 1)


        output_frame, results = detect_and_draw(yolomodel, frame, start_time)

        # Count the number of detected persons
        person_count += len([d for d in results.xyxy[0] if d[5] == 0])


    except:
        pass
    cv2.imshow('frame', frame)
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break