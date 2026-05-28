#!/usr/bin/env python3
import rospy
import sys
import select
import termios
import tty
import math

from quadrotor_msgs.msg import PositionCommand


class KeyboardTeleop:
    def __init__(self):
        rospy.init_node("keyboard_teleop_pos_cmd")

        # Step sizes (in meters / radians)
        self.step_xy = rospy.get_param("~step_xy", 0.5)      # move 0.5 m per key
        self.step_z  = rospy.get_param("~step_z", 0.5)       # move 0.5 m up/down
        self.step_yaw = rospy.get_param("~step_yaw", 0.001)  

        self.cmd_rate_hz = rospy.get_param("~rate", 20.0)

        # Target state in world frame (ENU):
        # x forward, y left, z up, yaw about z (rad, CCW from +x towards +y)
        self.x = 0.0
        self.y = 0.0
        self.z = 2.0   # start slightly above ground
        self.yaw = 0.0

        self.pub = rospy.Publisher("/planning/pos_cmd", PositionCommand, queue_size=10)

        # Save terminal settings so we can restore on exit
        self.settings = termios.tcgetattr(sys.stdin)

    def get_key(self, timeout):
        """
        Non-blocking key read, similar to teleop_twist_keyboard.
        Returns '' if no key was pressed within timeout.
        """
        dr, _, _ = select.select([sys.stdin], [], [], timeout)
        if dr:
            return sys.stdin.read(1)
        return ""

    def update_target_from_key(self, key):
        """
        Update target (x, y, z, yaw) based on pressed key.
        Motion is in the drone's body frame, mapped to world using current yaw.
        """
        if key == "w":
            # forward in body frame
            self.x += self.step_xy * math.cos(self.yaw)
            self.y += self.step_xy * math.sin(self.yaw)

        elif key == "s":
            # backward
            self.x -= self.step_xy * math.cos(self.yaw)
            self.y -= self.step_xy * math.sin(self.yaw)

        elif key == "d":
            # right (body frame)
            # right is -90 deg from forward in world frame
            self.x += self.step_xy * math.cos(self.yaw - math.pi / 2.0)
            self.y += self.step_xy * math.sin(self.yaw - math.pi / 2.0)

        elif key == "a":
            # left (body frame)
            self.x += self.step_xy * math.cos(self.yaw + math.pi / 2.0)
            self.y += self.step_xy * math.sin(self.yaw + math.pi / 2.0)

        elif key == " ":
            # up
            self.z += self.step_z

        elif key == ".":
            # down
            self.z -= self.step_z

        elif key == "k":
            # yaw left (CCW)
            self.yaw += self.step_yaw

        elif key == "l":
            # yaw right (CW)
            self.yaw -= self.step_yaw
            rospy.loginfo(f"yaw = {self.yaw}")

        elif key == "q":
            rospy.signal_shutdown("User requested shutdown")

        # clamp or wrap yaw if you like (optional)
        # self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

    def publish_pos_cmd(self):
        cmd = PositionCommand()
        cmd.header.stamp = rospy.Time.now()

        cmd.position.x = self.x
        cmd.position.y = self.y
        cmd.position.z = self.z

        # velocities / accelerations left at zero (we're doing position-only)
        cmd.velocity.x = 0.0
        cmd.velocity.y = 0.0
        cmd.velocity.z = 0.0
        cmd.acceleration.x = 0.0
        cmd.acceleration.y = 0.0
        cmd.acceleration.z = 0.0

        cmd.yaw = self.yaw              # radians
        cmd.yaw_dot = 0.0               # no spin command

        self.pub.publish(cmd)

    def run(self):
        rospy.loginfo("Keyboard teleop started.")
        rospy.loginfo("Controls: w/s/a/d = move, space = up, '.' = down, k/l = yaw, q = quit")

        rate = rospy.Rate(self.cmd_rate_hz)

        try:
            tty.setraw(sys.stdin.fileno())
            while not rospy.is_shutdown():
                key = self.get_key(1.0 / self.cmd_rate_hz)
                if key:
                    self.update_target_from_key(key)
                self.publish_pos_cmd()
                rate.sleep()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            rospy.loginfo("Keyboard teleop stopped, terminal restored.")


if __name__ == "__main__":
    node = KeyboardTeleop()
    node.run()
