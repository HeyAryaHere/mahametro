# This Is the testing file for the FastAPI server. It contains the code for the FastAPI server that will be used to serve the crowd counting model and provide real-time count information through a video stream.Dont forget to change the video path in the code to the path of the video file on your system.Also DONT USE THIS FILE FOR TESTING THE MODEL. USE test.html for testing the model. 
# -----------------------------------------------------------------------------------------------------------------------
'''TODO: 1.ADD the endpoint for the video stream
         2. ADD endpoints for Fall Detection 
         3. ADD Endpoints for line crossing detection
-------------------------------------------------------------------------------------------------------------------------
         '''

import cv2
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms
from PIL import Image
import numpy as np
import asyncio

app = FastAPI()

model = None
transform = None
video_path = r"C:\Users\suhru\OneDrive\Desktop\Github\mahametro\data\Crowding Near Ticketing line_1.avi"

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #only for html file testing test.html 
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

def load_model():
    # This function will be called once when the server starts
    '''
    -----------------------------------------------------------------------------------------------------------------------
    Load the model and the transform function for the model
    Args:None
    Returns:None
    -----------------------------------------------------------------------------------------------------------------------
    '''
    global model, transform
    from Model_files.crowd_counting.model_crowd import CSRNet

    model = CSRNet()
    checkpoint = torch.load('./Model_files/crowd_counting/weights.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@app.on_event("startup")
async def startup_event():
    load_model()

async def count_people(frames):
  
   '''
    -----------------------------------------------------------------------------------------------------------------------
   This function will count the number of people in the frame
    
    Args:The frames of the video from generate_frames
    
    Returns:Predicted count of people in the frame
    
    error:If the model is not loaded
    -----------------------------------------------------------------------------------------------------------------------
   '''
   with torch.no_grad():
        img = torch.stack([transform(Image.fromarray(frame)) for frame in frames])
        output = model(img)
        predicted_counts = output.detach().cpu().sum(dim=[1, 2, 3])
        return int(predicted_counts.mean().item())

async def generate_frames(video_path, interval=25):
   '''
   -----------------------------------------------------------------------------------------------------------------------
   
   This function will generate frames from the video
   
   Args:video_path:Path of the video, interval:Interval between frames to be processed
   
   Returns:Frames of the video (Split by interval)
   
   error:If the video is not loaded
   -----------------------------------------------------------------------------------------------------------------------
   '''
   
   cap = cv2.VideoCapture(video_path)
   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   frame_index = 0
    
   while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success:
            break
        
        count = await count_people([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])
        yield count
        
        frame_index += interval
        await asyncio.sleep(0.1)

class FrameCount(BaseModel):
    ''' This class will be used to return the count of people in the frame'''
    count: int

"""
-----------------------------------------------------------------------------------------------------------------------
Endpoint /count:
- Description: This endpoint returns the current count of people detected in the video stream.
  The count is obtained by processing the video frames using a pre-trained deep learning model for crowd counting.
- Method: GET
- Response: A JSON object containing the current count value.
- Example response:
    {
        "count": 25
    }
-----------------------------------------------------------------------------------------------------------------------
    """

@app.get("/count", response_model=FrameCount)
async def count_endpoint():
    count = await count_people([np.zeros((100, 100, 3), dtype=np.uint8)])  # Dummy frame to initialize
    return {"count": count}


"""
-----------------------------------------------------------------------------------------------------------------------
Endpoint /stream_count:
- Description: This endpoint streams the live count of people detected in the video stream.
  The count is continuously updated as new frames are processed, providing real-time information.
- Method: GET
- Response: A text/event-stream response containing the live count value.
  The response is formatted as a server-sent event (SSE) stream, with each event containing the current count value.
- Example response:
    data: 25
-----------------------------------------------------------------------------------------------------------------------
    """

@app.get("/stream_count")
async def stream_count():
    async def generate():
        async for count in generate_frames(video_path):
            yield f"data: {count}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


