import snap7
import struct
import argparse
import configparser
import os
import time
import cv2
from datetime import datetime
import threading

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
    

        
def main():
    while True:
        time.sleep(25)
    
        Data_trigger=readBool(db_number, input_offsets["Data_trigger"], 2)
    
        if Data_trigger == "True":
            capture_image_in_background()
    
        #OK_confirm=readBool(db_number, input_offsets["OK_confirm"], 2)
        #NOK_confirm=readBool(db_number, input_offsets["NOK_confirm"], 3)
    
        print('Data_trigger:',Data_trigger)
    
if __name__ == "__main__":
    main()

