import cv2  # For reading and displaying images
from ultralytics import YOLO  # YOLO model
import os  # For file handling
from pdf2image import convert_from_path  # For converting PDF pages to images

# Define constants
ENTITIES_COLORS = {
    "Caption": (191, 100, 21),
    "Footnote": (2, 62, 115),
    "Formula": (140, 80, 58),
    "List-item": (168, 181, 69),
    "Page-footer": (2, 69, 84),
    "Page-header": (83, 115, 106),
    "Picture": (255, 72, 88),
    "Section-header": (0, 204, 192),
    "Table": (116, 127, 127),
    "Text": (0, 153, 221),
    "Title": (196, 51, 2)
}
BOX_PADDING = 2

# Load the YOLO model
DETECTION_MODEL = YOLO("dla-model.pt")  # Make sure to use the correct path to your YOLO model
import matplotlib.pyplot as plt

def detect(image_path, output_dir="output", show_image=False):
    """
    Detect objects in an image, draw bounding boxes, and save the result.
    
    Args:
    - image_path (str): Path to the image file.
    - output_dir (str): Directory to save the output image.
    - show_image (bool): If True, display the image after detection using matplotlib.

    Returns:
    - image (numpy.ndarray): Image with bounding boxes drawn.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return None
    
    # Perform prediction on the image using the YOLO model
    results = DETECTION_MODEL.predict(source=image, conf=0.2, iou=0.8)
    boxes = results[0].boxes  # Get bounding boxes

    if len(boxes) == 0:
        print("No objects detected.")
        return image

    # Iterate through each detected bounding box
    for box in boxes:
        detection_class_conf = round(box.conf.item(), 2)  # Confidence score
        class_index = int(box.cls)
        cls = list(ENTITIES_COLORS)[class_index] if class_index < len(ENTITIES_COLORS) else "Unknown"
        
        # Log if class is not found in the dictionary
        if cls == "Unknown":
            print(f"Warning: Detected class index {class_index} is not mapped in ENTITIES_COLORS.")
        
        # Get start and end points of the bounding box
        start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
        end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

        # Draw bounding box with a color based on the class (default color for unknown classes)
        color = ENTITIES_COLORS.get(cls, (0, 255, 0))  # Green for unknown classes
        line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        image = cv2.rectangle(img=image, 
                              pt1=start_box, 
                              pt2=end_box,
                              color=color, 
                              thickness=line_thickness)

        # Create label text (class + confidence)
        text = f"{cls} {detection_class_conf}"
        
        # Get text dimensions for label background
        font_thickness = max(line_thickness - 1, 1)
        (text_w, text_h), _ = cv2.getTextSize(text=text, fontFace=2, fontScale=line_thickness / 3, thickness=font_thickness)
        
        # Draw background rectangle for the label
        image = cv2.rectangle(img=image,
                              pt1=(start_box[0], start_box[1] - text_h - BOX_PADDING * 2),
                              pt2=(start_box[0] + text_w + BOX_PADDING * 2, start_box[1]),
                              color=color,
                              thickness=-1)

        # Put text (label) on top of the background rectangle
        start_text = (start_box[0] + BOX_PADDING, start_box[1] - BOX_PADDING)
        image = cv2.putText(img=image, text=text, org=start_text, fontFace=0, color=(255, 255, 255), fontScale=line_thickness / 3, thickness=font_thickness)
    
    # Save the image with bounding boxes
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")
    
    # Optionally show the image using matplotlib
    if show_image:
        # Convert BGR image (OpenCV default) to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')  # Hide axes
        plt.show()

    return image

def process_pdf(pdf_path, output_dir="output", dpi=300):
    """
    Convert each page of a PDF to an image, then run detection on each image.
    
    Args:
    - pdf_path (str): Path to the PDF file.
    - output_dir (str): Directory to save output images.
    - dpi (int): Dots per inch for PDF-to-image conversion (default: 300).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=dpi)
    
    # Process each page as an image
    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        page.save(image_path, "JPEG")
        print(f"Processing page {i+1}/{len(pages)}")
        
        # Detect objects in the image
        detect(image_path, output_dir=output_dir, show_image=False)

if __name__ == "__main__":
    # Example usage for PDFs
    input_pdf_path = "test.pdf"
    process_pdf(input_pdf_path, output_dir="output", dpi=300)
