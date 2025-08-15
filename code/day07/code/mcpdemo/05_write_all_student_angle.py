#!/usr/bin/env python
# -*- coding: utf-8 -*-
import serial
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- ServoController Class (adapted from previous scripts) ---

class ServoController:
    def __init__(self, port="COM6", baudrate=1000000, timeout=0.1):
        self.port_name = port
        self.serial_port = None
        # Constants
        self.ADDR_GOAL_POSITION = 42
        self.ADDR_TORQUE_ENABLE = 40
        self.INST_WRITE = 3
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

    def _write_register(self, servo_id, address, value, size=2):
        params = [address]
        if size == 1:
            params.append(value & 0xFF)
        elif size == 2:
            params.extend([value & 0xFF, (value >> 8) & 0xFF])
        else:
            return False
        
        # For write operations, we often don't need to wait for a response
        # unless we need to confirm the write was successful.
        # Removing the read can make operations faster, especially for sliders.
        if not self._send_packet(servo_id, self.INST_WRITE, params):
            return False
        
        # We assume success if the packet was sent.
        return True

    def enable_torque(self, servo_id):
        # This is a fire-and-forget write. We don't check the response.
        return self._write_register(servo_id, self.ADDR_TORQUE_ENABLE, 1, size=1)

    def set_servo_angle(self, servo_id, angle):
        """Sets the servo to a specific angle (-90 to 90 degrees)."""
        # Map angle (-90 to 90) to position (1024 to 3072)
        position = int(((angle + 90.0) / 180.0) * (3072.0 - 1024.0) + 1024.0)
        # Clamp the value to be safe
        position = max(1024, min(3072, position))
        
        return self._write_register(servo_id, self.ADDR_GOAL_POSITION, position, size=2)

# --- Main function with GUI ---

def main():
    """Main function - creates a GUI to control all student servos."""
    # Configuration
    STUDENT_PORT = "COM6"  # Adjust COM port if necessary
    BAUDRATE = 1000000
    SERVO_IDS = list(range(1, 7)) # Servo IDs 1-6
    
    controller = None
    try:
        controller = ServoController(port=STUDENT_PORT, baudrate=BAUDRATE)

        # Enable torque for all servos first
        print("Enabling torque for all servos...")
        for servo_id in SERVO_IDS:
            controller.enable_torque(servo_id)
            time.sleep(0.05) # Small delay between commands
        print("Torque enabled for all servos.")

        # Setup matplotlib GUI
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(left=0.25, bottom=0.1, top=0.9, right=0.9)
        fig.suptitle('Student Servo Angle Control')

        sliders = []
        for i, servo_id in enumerate(SERVO_IDS):
            ax_slider = plt.axes([0.25, 0.1 + i * 0.12, 0.65, 0.05])
            slider = Slider(
                ax=ax_slider,
                label=f'Servo {servo_id}',
                valmin=-90,
                valmax=90,
                valinit=0,
                valfmt='%0.1f°'
            )
            sliders.append(slider)

        def update_factory(servo_id):
            def update(val):
                angle = sliders[servo_id-1].val
                if not controller.set_servo_angle(servo_id, angle):
                    print(f"\rFailed to set angle for servo {servo_id}", end="")
                else:
                    # Clear previous failure message if any
                    print(f"\rServo {servo_id} angle set to {angle:.1f}°   ", end="")
            return update

        for i, servo_id in enumerate(SERVO_IDS):
            sliders[i].on_changed(update_factory(servo_id))

        print("\nGUI is ready. Drag sliders to control servos. Close the window to exit.")
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