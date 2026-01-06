from flask import Flask, render_template, jsonify
import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import send_from_directory
import snap7
import argparse
import configparser
import time


############################################################################################################################################

parser = argparse.ArgumentParser()
config_file="C:\\2023\\Python_Training\\PLC_label\\config_ini.txt"
section="MATS"
config=configparser.ConfigParser()
config.read(config_file)


get_CPU_host=config.get(section,'CPU_host')
get_CPU_slot=int(config.get(section,'CPU_slot'))
get_CPU_rack=int(config.get(section,'CPU_rack'))
get_db_number=int(config.get(section,'db_number'))
get_start_address=int(config.get(section,'start_address'))
get_length=int(config.get(section,'length'))
get_start_offset=int(config.get(section,'start_offset'))

plc = snap7.client.Client()
plc.connect(get_CPU_host, get_CPU_rack, get_CPU_slot)

db_number =  int(get_db_number)
start_offset =  int(get_start_offset)
value = 1

start_address = 0 
length = 12

input_offsets={"Data_trigger":2,"OK_confirm":0,"NOK_confirm":0}
reading1 = plc.db_read(db_number, start_address, length)

############################################################################################################################################


# Define the path for the Excel file
excel_file_path = 'D:/ANAND Project/MATS Visco Clutch Rolling Operation/Final Model/Trial UI/Result Excel File/processing_results.xlsx'

# Create the Excel file with the necessary columns if it doesn't exist
if not os.path.exists(excel_file_path):
    df = pd.DataFrame(columns=['Serial No', 'Image name', 'Product Status','Timestamp'])
    df.to_excel(excel_file_path, index=False)
    

# Load the pre-trained models
rolling_model = load_model('rollingcheck_model.h5')

# Coordinates for each component
rolling_coordinates = {'x': 1680.0, 'y': 2560.0, 'width': 230.0, 'height': 1180.0}


app = Flask(__name__)

# Global variable to store the latest image path
dynamic_latest_image_path = ""
absolute_latest_image_path =""

###############################################################################################################################
#Home page Code 
# Define a route to serve the HTML file at the root URL
@app.route('/')
def index():
    return render_template('Trial.html')  # Ensure you have a 'Trial.html' file in a 'templates' folder

#################################################################################################################################

# Route to fetch the latest image from a specified folder
@app.route('/latest-image')
def latest_image():
        
    def writeBool(db_number, start_offset, bit_offset, value):
        reading1 = plc.db_read(db_number, start_offset, 1)    # (db number, start offset, read 1 byte)
        snap7.util.set_bool(reading1, 0, bit_offset, value)   # (value 1= true;0=false) (bytearray_: bytearray, byte_index: int, bool_index: int, value: bool)
        plc.db_write(db_number, start_offset, reading1)       #  write back the bytearray and now the boolean value is changed in the PLC.
        ##print('DB Number: ' + str(db_number) + ' Bit: ' + str(start_offset) + '.' + str(bit_offset) + ' Value: ' + str(a))
        return None

    def readBool(db_number, start_offset, bit_offset):
        reading1 = plc.db_read(db_number, start_offset, 1)  
        a = snap7.util.get_bool(reading1, 0, bit_offset)
        ##print('DB Number: ' + str(db_number) + ' Bit: ' + str(start_offset) + '.' + str(bit_offset) + ' Value: ' + str(a))
        return str(a)

    def capture_image_in_background():
        save_directory = r"C:/2024/AI/Final Model Visco Rolling/static/images/"
        os.makedirs(save_directory, exist_ok=True)
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        ret, frame = cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"captured_image_{timestamp}.jpg"
            save_path = os.path.join(save_directory, filename)
            cv2.imwrite(save_path, frame)
            print(f"Image captured and saved to {save_path}")
        else:
            print("Error: Failed to capture image.")

        cap.release()
        
    
    Data_trigger=readBool(db_number, input_offsets["Data_trigger"], 2)
    
    if Data_trigger == "True":
        capture_image_in_background()

    time.sleep(3)
    
    global dynamic_latest_image_path  # Declare the global variable
    global absolute_latest_image_path
    
    IMAGE_FOLDER = 'D:/ANAND Project/MATS Visco Clutch Rolling Operation/Final Model/Trial UI/static/images/'  # Update with your actual image folder path
    files = sorted(
        (f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))),
        key=lambda f: os.path.getmtime(os.path.join(IMAGE_FOLDER, f)),
        reverse=True
    )
    if files:
        dynamic_latest_image_path = f"/static/images/{files[0]}" #Dynamic path can not be used for the processing the image
        absolute_latest_image_path = os.path.join(IMAGE_FOLDER, files[0]) #Absolute path
        print("Latest Image Path updated in function:", dynamic_latest_image_path)  # Print the latest image path inside the function
        return jsonify({"imagePath": dynamic_latest_image_path})
    
    dynamic_latest_image_path = ""  # Set to empty if no images found
    print("No images found in the directory.")
    return jsonify({"imagePath": ""})

###############################################################################################################################

last_processed_image = ""


@app.route('/process-latest-image')
def process_latest_image():
    global absolute_latest_image_path, last_processed_image  # Access the global variable
            
     # Only proceed if there is a new image
    if absolute_latest_image_path and absolute_latest_image_path != last_processed_image:
        # Update the last processed image path
        last_processed_image = absolute_latest_image_path

        
        # Process the image using latest_image_path
        print("Processing image at path:", absolute_latest_image_path)
        
        output_folder = 'D:/ANAND Project/MATS Visco Clutch Rolling Operation/Final Model/Trial UI/Result/'
        img = cv2.imread(absolute_latest_image_path)
        image_name = os.path.basename(absolute_latest_image_path)
        
        # Prediction and annotation functions (from your code)
        def predict_and_draw(img, model, coordinates, label):
            x, y, width, height = int(coordinates['x']), int(coordinates['y']), int(coordinates['width']), int(coordinates['height'])
            roi = img[y:y + height, x:x + width]
            roi_resized = cv2.resize(roi, (224, 224)) / 255.0  # Normalize

            prediction = model.predict(np.expand_dims(roi_resized, axis=0))
            confidence = np.max(prediction) * 100
            predicted_class = np.argmax(prediction, axis=1)[0]
            display_text = f'{label}: {"Okay" if predicted_class == 1 else "Not Okay"} ({confidence:.2f}%)'

            color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
            x = x - 300
            y = y - 800
            width = width + 500
            height = height + 500
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 19)
            cv2.putText(img, display_text, (x - 20, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 7)
            
            return predicted_class

        
        # Apply the models to the image and get results for each component
        rolling_result = "Okay" if predict_and_draw(img, rolling_model, rolling_coordinates, 'Rolling') == 1 else "Not Okay"
            
            
        # Save the processed image with the same name as the original image
        output_path = os.path.join(output_folder, os.path.basename(absolute_latest_image_path))
        cv2.imwrite(output_path, img)
        
        # Load the existing Excel file and append the new row
        df = pd.read_excel(excel_file_path)
        new_row = {
            'Serial No': len(df) + 1,
            'Image name': image_name,
            'Product Status': rolling_result,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format as desired
        }
        df = df._append(new_row, ignore_index=True)
        
        # Save the updated DataFrame back to the Excel file
        df.to_excel(excel_file_path, index=False)
                           
        print("Processed image at path:", absolute_latest_image_path)
        # Add any processing logic here
        # Determine overall result based on individual component results

        overall_status = "Okay" if all(result == "Okay" for result in [rolling_result]) else "Not Okay"

        os.remove(absolute_latest_image_path)
        
        # Include the overall status in the response
        return jsonify({"message": f"Processed the latest image at path: {absolute_latest_image_path}", "overallStatus": overall_status})
    else:
        return "No latest image available to process."

#########################################################################################################################################
        
# Route to fetch the latest processed image from the result folder
@app.route('/latest-result-image')
def latest_result_image():
    result_folder = 'D:/ANAND Project/MATS Visco Clutch Rolling Operation/Final Model/Trial UI/Result/'  # Path to the result folder
    files = sorted(
        (f for f in os.listdir(result_folder) if os.path.isfile(os.path.join(result_folder, f))),
        key=lambda f: os.path.getmtime(os.path.join(result_folder, f)),
        reverse=True
    )
    if files:
        latest_result_image_path = f"/result/{files[0]}"
        print("Latest Result Image Path:", latest_result_image_path)
        return jsonify({"imagePath": latest_result_image_path})
    
    return jsonify({"imagePath": ""})

@app.route('/result/<path:filename>')
def serve_result_image(filename):
    result_directory = "D:/ANAND Project/MATS Visco Clutch Rolling Operation/Final Model/Trial UI/Result/"
    return send_from_directory(result_directory, filename)


if __name__ == '__main__':
    app.run(debug=False)





##############################################################################################################################################
##############################################################################################################################################
'''
from flask import Flask, render_template, jsonify
import os
import threading
import time
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import send_from_directory

# Define the path for the Excel file
excel_file_path = 'D:/ANAND Project/MATS Visco Clutch Rolling Operation/Final Model/Trial UI/Result Excel File/processing_results.xlsx'

# Create the Excel file with the necessary columns if it doesn't exist
if not os.path.exists(excel_file_path):
    df = pd.DataFrame(columns=['Serial No', 'Image name', 'Product Status','Timestamp'])
    df.to_excel(excel_file_path, index=False)
    

# Load the pre-trained models
rolling_model = load_model('rollingcheck_model.h5')

# Coordinates for each component
rolling_coordinates = {'x': 1680.0, 'y': 2560.0, 'width': 230.0, 'height': 1180.0}


app = Flask(__name__)

# Global variable to store the latest image path
dynamic_latest_image_path = ""
absolute_latest_image_path =""

###############################################################################################################################
#Home page Code 
# Define a route to serve the HTML file at the root URL
@app.route('/')
def index():
    return render_template('Trial.html')  # Ensure you have a 'Trial.html' file in a 'templates' folder

#################################################################################################################################

# Route to fetch the latest image from a specified folder
@app.route('/latest-image')
def latest_image():
    global dynamic_latest_image_path  # Declare the global variable
    global absolute_latest_image_path
    
    IMAGE_FOLDER = 'D:/ANAND Project/MATS Visco Clutch Rolling Operation/Final Model/Trial UI/static/images/'  # Update with your actual image folder path
    files = sorted(
        (f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))),
        key=lambda f: os.path.getmtime(os.path.join(IMAGE_FOLDER, f)),
        reverse=True
    )
    if files:
        dynamic_latest_image_path = f"/static/images/{files[0]}" #Dynamic path can not be used for the processing the image
        absolute_latest_image_path = os.path.join(IMAGE_FOLDER, files[0]) #Absolute path
        print("Latest Image Path updated in function:", dynamic_latest_image_path)  # Print the latest image path inside the function
        return jsonify({"imagePath": dynamic_latest_image_path})
    
    dynamic_latest_image_path = ""  # Set to empty if no images found
    print("No images found in the directory.")
    return jsonify({"imagePath": ""})

###############################################################################################################################

last_processed_image = ""


@app.route('/process-latest-image')
def process_latest_image():
    global absolute_latest_image_path, last_processed_image  # Access the global variable
            
     # Only proceed if there is a new image
    if absolute_latest_image_path and absolute_latest_image_path != last_processed_image:
        # Update the last processed image path
        last_processed_image = absolute_latest_image_path

        
        # Process the image using latest_image_path
        print("Processing image at path:", absolute_latest_image_path)
        
        output_folder = 'D:/ANAND Project/MATS Visco Clutch Rolling Operation/Final Model/Trial UI/Result/'
        img = cv2.imread(absolute_latest_image_path)
        image_name = os.path.basename(absolute_latest_image_path)
        
        # Prediction and annotation functions (from your code)
        def predict_and_draw(img, model, coordinates, label):
            x, y, width, height = int(coordinates['x']), int(coordinates['y']), int(coordinates['width']), int(coordinates['height'])
            roi = img[y:y + height, x:x + width]
            roi_resized = cv2.resize(roi, (224, 224)) / 255.0  # Normalize

            prediction = model.predict(np.expand_dims(roi_resized, axis=0))
            confidence = np.max(prediction) * 100
            predicted_class = np.argmax(prediction, axis=1)[0]
            display_text = f'{label}: {"Okay" if predicted_class == 1 else "Not Okay"} ({confidence:.2f}%)'

            color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
            x = x - 300
            y = y - 800
            width = width + 500
            height = height + 500
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 19)
            cv2.putText(img, display_text, (x - 20, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 7)
            
            return predicted_class

        
        # Apply the models to the image and get results for each component
        rolling_result = "Okay" if predict_and_draw(img, rolling_model, rolling_coordinates, 'Rolling') == 1 else "Not Okay"
            
            
        # Save the processed image with the same name as the original image
        output_path = os.path.join(output_folder, os.path.basename(absolute_latest_image_path))
        cv2.imwrite(output_path, img)
        
        # Load the existing Excel file and append the new row
        df = pd.read_excel(excel_file_path)
        new_row = {
            'Serial No': len(df) + 1,
            'Image name': image_name,
            'Product Status': rolling_result,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format as desired
        }
        df = df._append(new_row, ignore_index=True)
        
        # Save the updated DataFrame back to the Excel file
        df.to_excel(excel_file_path, index=False)
                           
        print("Processed image at path:", absolute_latest_image_path)
        # Add any processing logic here
        # Determine overall result based on individual component results

        overall_status = "Okay" if all(result == "Okay" for result in [rolling_result]) else "Not Okay"

        os.remove(absolute_latest_image_path)
        
        # Include the overall status in the response
        return jsonify({"message": f"Processed the latest image at path: {absolute_latest_image_path}", "overallStatus": overall_status})
    else:
        return "No latest image available to process."

#########################################################################################################################################
        
# Route to fetch the latest processed image from the result folder
@app.route('/latest-result-image')
def latest_result_image():
    result_folder = 'D:/ANAND Project/MATS Visco Clutch Rolling Operation/Final Model/Trial UI/Result/'  # Path to the result folder
    files = sorted(
        (f for f in os.listdir(result_folder) if os.path.isfile(os.path.join(result_folder, f))),
        key=lambda f: os.path.getmtime(os.path.join(result_folder, f)),
        reverse=True
    )
    if files:
        latest_result_image_path = f"/result/{files[0]}"
        print("Latest Result Image Path:", latest_result_image_path)
        return jsonify({"imagePath": latest_result_image_path})
    
    return jsonify({"imagePath": ""})

@app.route('/result/<path:filename>')
def serve_result_image(filename):
    result_directory = "D:/ANAND Project/MATS Visco Clutch Rolling Operation/Final Model/Trial UI/Result/"
    return send_from_directory(result_directory, filename)


if __name__ == '__main__':
    app.run(debug=False)
'''