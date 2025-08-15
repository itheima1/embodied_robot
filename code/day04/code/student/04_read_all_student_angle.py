#!/usr/bin/env python
# -*- coding: utf-8 -*-
import serial
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- ServoController Class from 03_read_student_angle.py ---

class ServoController:
    def __init__(self, port="COM4", baudrate=1000000, timeout=0.1):
        self.port_name = port
        self.serial_port = None
        # Constants
        self.ADDR_PRESENT_POSITION = 56
        self.INST_READ = 2
        self.COMM_SUCCESS = 0
        self.COMM_RX_TIMEOUT = -6
        self.COMM_RX_CORRUPT = -7
        
        try:
            self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            time.sleep(0.1)
            self.serial_port.reset_input_buffer()
            print(f"Successfully opened port {port}.")
        except serial.SerialException as e:
            print(f"Fatal: Could not open port {port}: {e}")
            sys.exit(1)

    def close(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print(f"Port {self.port_name} closed.")

    def _calculate_checksum(self, data):
        return (~sum(data)) & 0xFF

    def _send_packet(self, servo_id, instruction, parameters=None):
        if not self.serial_port or not self.serial_port.is_open:
            return False
        if parameters is None:
            parameters = []
        length = len(parameters) + 2
        packet_core = [servo_id, length, instruction] + parameters
        checksum = self._calculate_checksum(packet_core)
        packet = bytes([0xFF, 0xFF] + packet_core + [checksum])
        try:
            self.serial_port.reset_input_buffer()
            self.serial_port.write(packet)
            self.serial_port.flush()
            return True
        except Exception:
            return False

    def _read_packet(self):
        start_time = time.time()
        packet = []
        while (time.time() - start_time) < self.serial_port.timeout:
            if self.serial_port.in_waiting > 0:
                byte = self.serial_port.read(1)
                if not byte: continue
                byte = byte[0]

                if not packet and byte != 0xFF:
                    continue
                
                packet.append(byte)

                if len(packet) >= 2 and packet[-2:] == [0xFF, 0xFF]:
                    if len(packet) > 2:
                        packet = [0xFF, 0xFF]
                    continue

                if len(packet) > 4:
                    pkt_len = packet[3]
                    if len(packet) == pkt_len + 4:
                        core_data = packet[2:-1]
                        calculated_checksum = self._calculate_checksum(core_data)
                        if calculated_checksum == packet[-1]:
                            return self.COMM_SUCCESS, packet[4], packet[5:-1]
                        else:
                            return self.COMM_RX_CORRUPT, 0, []
        return self.COMM_RX_TIMEOUT, 0, []

    def get_servo_angle(self, servo_id):
        if not self._send_packet(servo_id, self.INST_READ, [self.ADDR_PRESENT_POSITION, 2]):
            return None

        result, error, data = self._read_packet()

        if result != self.COMM_SUCCESS or error != 0:
            return None
        
        if data and len(data) >= 2:
            position = data[0] | (data[1] << 8)
            angle = ((position - 1024.0) / (3072.0 - 1024.0)) * 180.0 - 90.0
            angle = max(-90.0, min(90.0, angle))
            return angle
        
        return None

# --- Main function adapted from 02_read_all_teacher_angle.py ---

def main():
    """Main function - reads and visualizes all student servo angles."""
    # Configuration
    STUDENT_PORT = "COM6"  # Adjust COM port if necessary
    BAUDRATE = 1000000
    SERVO_IDS = list(range(1, 7)) # Servo IDs 1-6
    
    controller = None
    try:
        controller = ServoController(port=STUDENT_PORT, baudrate=BAUDRATE)
        
        # Setup matplotlib for plotting
        fig, ax = plt.subplots()
        joint_names = [f'Servo {i}' for i in SERVO_IDS]
        y_pos = np.arange(len(joint_names))
        bars = ax.barh(y_pos, [0] * len(SERVO_IDS), align='center')
        ax.set_yticks(y_pos, labels=joint_names)
        ax.invert_yaxis()  # To display servo 1 at the top
        ax.set_xlabel('Angle (°)')
        ax.set_title('Real-time Student Joint Angles')
        ax.set_xlim(-100, 100) # Range is -90 to 90, give a little buffer

        text_labels = [ax.text(0, i, '0.0°', va='center') for i in range(len(SERVO_IDS))]

        def update(frame):
            """Function to update the plot for the animation."""
            angles = []
            for servo_id in SERVO_IDS:
                angle = controller.get_servo_angle(servo_id)
                angles.append(angle if angle is not None else 0) # Show 0 if read fails
            
            for i, (bar, angle_val) in enumerate(zip(bars, angles)):
                bar.set_width(angle_val)
                bar.set_color('steelblue' if angle_val >= 0 else 'lightcoral')
                
                text_labels[i].set_position((angle_val + 5 if angle_val >= 0 else angle_val - 5, i))
                text_labels[i].set_text(f"{angle_val:.1f}°")
                text_labels[i].set_ha('left' if angle_val >= 0 else 'right')

            return list(bars) + text_labels

        # Create and run the animation
        ani = animation.FuncAnimation(fig, update, blit=True, interval=50, cache_frame_data=False)

        plt.show()

    except KeyboardInterrupt:
        print("\nStopping program.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if controller:
            controller.close()
        print("\nProgram finished.")

if __name__ == '__main__':
    main()