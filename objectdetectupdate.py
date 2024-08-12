import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

# Load the pre-trained model and feature extractor
model_name = "facebook/detr-resnet-50"
feature_extractor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preprocess the frame
    inputs = feature_extractor(images=frame_rgb, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process outputs
    logits = outputs.logits
    boxes = outputs.pred_boxes

    # Convert boxes to coordinates in the original frame size
    img_height, img_width, _ = frame.shape
    boxes = boxes * torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float)

    # Draw bounding boxes on the frame
    for logit, box in zip(logits[0], boxes[0]):
        if logit.max() > 0.5:  # confidence threshold
            label = logit.argmax(-1).item()
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {label}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
