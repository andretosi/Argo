from enum import IntEnum

class STInstruction(IntEnum):
    PING = 0x01                                     # 1 Byte
    READ_DATA = 0x02                                # 1 Byte
    WRITE_DATA = 0x03                               # 1 Byte
    REGWRITE_DATA = 0x04                            # 1 Byte
    ACTION = 0x05                                   # 1 Byte
    SYNC_READ_DATA = 0x82                           # 1 Byte
    SYNC_WRITE_DATA = 0x83                          # 1 Byte
    RESET = 0x0A                                    # 1 Byte

class STEPROMConfiguration(IntEnum):
    SERVO_ID = 0x05                                 # 1 Byte
    BAUD_RATE = 0x06                                # 1 Byte
    RESPONSE_STATUS_LEVEL = 0x08                    # 1 Byte
    MINIMUM_ANGLE_LIMIT = 0x09                      # 2 Bytes
    MAXIMUM_ANGLE_LIMIT = 0x0B                      # 2 Bytes
    MAXIMUM_TEMPERATURE = 0x0D                      # 1 Byte
    MAXIMUM_INPUT_VOLTAGE = 0x0E                    # 1 Byte
    MINIMUM_INPUT_VOLTAGE = 0x0F                    # 1 Byte
    MAXIMUM_TORQUE = 0x10                           # 2 Bytes
    PHASE = 0x12                                    # 1 Byte
    UNLOAD_CONDITION = 0x13                         # 1 Byte
    LED_ALARM_CONDITION = 0x14                      # 1 Byte
    POSITION_LOOP_P_PROPORTIONAL_COEFFICIENT = 0x15 # 1 Byte
    POSITION_LOOP_D_DERIVATIVE_COEFFICIENT = 0x16   # 1 Byte
    POSITION_LOOP_I_INTEGRAL_COEFFICIENT = 0x17     # 1 Byte
    MINIMUM_STARTING_TORQUE = 0x18                  # 1 Byte
    INTEGRAL_LIMIT_VALUE = 0x19                     # 1 Byte
    POSITIVE_DEAD_ZONE = 0x1A                       # 1 Byte
    NEGATIVE_DEAD_ZONE = 0x1B                       # 1 Byte
    PROTECTION_CURRENT = 0x1C                       # 2 Bytes
    ANGLE_RESOLUTION = 0x1E                         # 1 Byte
    POSITION_OFFSET = 0x1F                          # 2 Bytes
    OPERATING_MODE = 0x21                           # 1 Byte
    HOLDING_TORQUE = 0x22                           # 1 Byte
    PROTECTION_TIME = 0x23                          # 1 Byte
    OVERLOAD_TORQUE = 0x24                          # 1 Byte
    SPEED_CLOSED_LOOP_P_PROPORTIONAL_COEFFICIENT = 0x25 # 1 Byte
    OVERCURRENT_PROTECTION_TIME = 0x26             # 1 Byte
    SPEED_CLOSED_LOOP_I_INTEGRAL_COEFFICIENT = 0x27 # 1 Byte

class STSRAMControl(IntEnum):
    TORQUE_SWITCH = 0x28                            # 1 Byte
    ACCELERATION = 0x29                             # 1 Byte
    TARGET_POSITION = 0x2A                          # 2 Bytes
    PWM_OPEN_LOOP_SPEED = 0x2C                      # 2 Bytes
    RUNNING_SPEED = 0x2E                            # 2 Bytes
    TORQUE_LIMIT = 0x30                             # 2 Bytes
    LOCK_FLAG = 0x37                                # 1 Byte

class STSRAMFeedback(IntEnum):
    CURRENT_POSITION = 0x38                         # 2 Bytes
    CURRENT_SPEED = 0x3A                            # 2 Bytes
    CURRENT_LOAD = 0x3C                             # 2 Bytes
    CURRENT_VOLTAGE = 0x3E                          # 1 Byte
    CURRENT_TEMPERATURE = 0x3F                      # 1 Byte
    ASYNCHRONOUS_WRITE_FLAG = 0x40                  # 1 Byte
    SERVO_STATUS = 0x41                             # 1 Byte
    MOVING_FLAG = 0x42                              # 1 Byte
    CURRENT_TARGET_POSITION = 0x43                  # 2 Bytes
    CURRENT_CURRENT = 0x45                          # 2 Bytes

class STServoStatus(IntEnum):
    OK = 0x00                                       # 1 Byte
    VOLTAGE_ERROR = 1 << 0                          # 1 Byte
    SENSOR_ERROR = 1 << 1                           # 1 Byte
    OVERTEMP_ERROR = 1 << 2                         # 1 Byte
    OVERCURRENT_ERROR = 1 << 3                      # 1 Byte
    OVERLOAD_ERROR = 1 << 5                         # 1 Byte