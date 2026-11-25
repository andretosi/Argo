import serial
import ST3235_registers
from time import sleep

"""
this file contains some low level features to interact with the servos.
this code isn't by any mean complete and can be used only with extreeme caution(at this stage).
S.A.
"""


class serial_handler:
    
    ser:serial
    devicename:str
    baudrate:int
    MAX_TRYS:int = 12
    
    def __init__(self,devicename = "dev/ttyACM0",baudrate:int = 1000000)->None: #initialize serial port
        self.devicename=devicename
        self.baudrate=baudrate
        self.ser=serial.Serial(port=self.devicename,baudrate=self.baudrate,timeout=0.5)
    
    """-----------------------------sending and variations--------------------------------------"""
    
    def send_(self,msg:list[int],id:int)->None:
        """
        sends msg to the servo id.
        msg must be a list of ints and contain instruction and parameters
        """
        
        lenght:int=len(msg)+1
        chksum=~(id+lenght+sum(msg))&0xff
        
        sendble:list[int]=[0xff,0xff]   #header
        sendble.append(id)              #id
        sendble.append(lenght)          #lenght
        for x in msg:
            sendble.append(x)           #comand + parameters
        sendble.append(chksum)          #chksum
        
        for x in sendble: print(hex(x)) #printing sended message usefull for debug
        self.ser.write(bytes(sendble,))
        
    def send_brodcast(self,msg:list[int])->None:
        """
        sends msg to every servo.
        msg must be a list of ints and contain instruction and parameters.
        brodcasts calls do not (usually) generate answeres.
        """
        
        return self.send_(msg=msg,id=0xfe)
        
    """------------reciving, variations and managing of related exeptions------------------------"""
    
    def receive_(self)->list[int,list[int]]:
        """
        takes no parameters and outputs the id of the responce and a list of int8 as the parameters. 
        if your expected responce has vaues of multiple bytes you must manage it.
        this method will respond None if a comunication error has occurred.
        this method will call a hardwere_hazard function if the servo signals.
        """
        
        #header heandling
        first:bytes=self.safe_read_bus(1)
        second:bytes=b''
        trys:int =0
        
        while(not (first==second and second==b'\xFF') ):
            second=first
            first=self.safe_read_bus(1)
            trys+=1
            if(trys==self.MAX_TRYS): raise TimeoutError
            
        
        #takes id,length and status code
        id:int= int.from_bytes(self.safe_read_bus(1))
        length:int=int.from_bytes(self.safe_read_bus(1))
        status:int = int.from_bytes(self.safe_read_bus(1))
        
        #if status is not 0 the servo is signaling a problem, the hardwere might be in danger https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation#2-special-byte-explanation
        if (bool(status)):self.harwere_hazard()
        
        #takes the parameters
        params:list[int]
        params=list(self.safe_read_bus(length-2))
        
        # validate the check sum, in case it differs it is a sign of comm error
        chksum:int =~(id+length+status+sum(params))&0xff
        chksum_res:int =int.from_bytes(self.safe_read_bus(1))
        if (chksum!=chksum_res):
            return None
            
        return id, params
        
    
    def harwere_hazard(self,id:int,status:int): #signals user to disconnect and stops the code 
        """
        this function should be a big red button. to do: find a better way to handle it
        """
        
        print("DANGER DANGER DANGER disconnect now!"+id+"reason:"+status)   
        raise TimeoutError 
            
            
    def safe_read_bus(self,n:int)->bytes: #handles eventual timeouts 
        return self.ser.read(n) #to do: make it safe
    
    
    """----------------------------------debug------------------------------------"""
    
    def print_buss(self): #function for debugghing purpous: prints everythin in the bus as long as there is something
            sleep(0.1)
            msg:int = self.ser.read(1)
            
            while(msg!=b''):
                print(int.from_bytes(msg))
                msg:int = self.ser.read(1)
