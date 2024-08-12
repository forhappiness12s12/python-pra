import cv2
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

# Load the pre-trained model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

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
    
    # Get predictions
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Display the resulting frame
    cv2.putText(frame, f"Predicted class: {predicted_class_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
