import serial
import ST3235_registers

"""
this file contains some low level features to interact with the servos.
this code isn't by any mean complete and shoud not be used (at this stage).
S.A.
"""


class serial_handler:
    
    ser:serial
    devicename:str
    baudrate:int
    
    def __init__(self,devicename = "dev/ttyACM0",baudrate:int = 1000000)->None:
        self.devicename=devicename
        self.baudrate=baudrate
        self.ser=serial.Serial(port=self.devicename,baudrate=self.baudrate,timeout=50)
    
    def send_(self,msg:list[int],id:int)->None:
        """msg must be a list of ints and contain the instruction and the parameters"""
        
        lenght:int=len(msg)+1
        chksum=~(id+lenght+sum(msg))&0xff
        
        sendble:list[int]=[0xff,0xff]   #header
        sendble.append(id)              #id
        sendble.append(lenght)          #lenght
        for x in msg:
            sendble.append(x)           #comand + parameters
        sendble.append(chksum)          #chksum
        
        #for x in sendble: print(hex(x)) #printing sended message usefull for debug
        self.ser.write(bytes(sendble,))
        
    def send_brodcast(self,msg:list[int])->None:
        return self.send_(msg=msg,id=0xfe)

