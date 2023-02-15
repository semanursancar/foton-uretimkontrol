import cv2
import pandas as pd
from pytesseract import image_to_data
import pytesseract
from flask import Flask, request, render_template
from io import BytesIO
import pandas as pd
import numpy as np
import os

def process_images(image_files):
    # Create an empty DataFrame to hold the OCR results
    values = pd.DataFrame()
    # Loop through each image file
    for image_file in image_files:
        # Load image with OpenCV
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        # # Run OCR on the processed image

        path = os.environ['tesseract_path']

        pytesseract.pytesseract.tesseract_cmd = r'{}\tesseract.exe'.format(path)


        # Resimdeki metinleri algıla
        data = image_to_data(img, output_type='data.frame')
        # DataFrame'e dönüştür
        df = pd.DataFrame(data)
        values_this_image = pd.DataFrame(df["text"])
        # Drop rows with missing values
        values_this_image.dropna(axis=0, inplace=True)
        # Remove rows containing invalid characters
        values_this_image = values_this_image[~values_this_image['text'].str.contains("K")]
        values_this_image = values_this_image[~values_this_image['text'].str.contains("V")]
        values_this_image = values_this_image[~values_this_image['text'].str.contains("M")]
        values_this_image = values_this_image[~values_this_image['text'].str.contains(" ")]
        # Add OCR results as a new column to the DataFrame
        values[image_file.filename] = values_this_image["text"]
    # Reset the index of the DataFrame
    values.reset_index(drop=True, inplace=True)
    # Return the DataFrame
    return values



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded images
        image_files = request.files.getlist("file")
        # Process images with OCR
        values = process_images(image_files)
        # Create HTML table from results
        table = values.to_html(classes='table table-striped')
        # Return HTML response
        return render_template("result.html", table=table)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
