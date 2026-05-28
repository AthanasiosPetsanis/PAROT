#!/usr/bin/env python3

import rospy
import math
from quadrotor_msgs.msg import PositionCommand
from airsim_ros_pkgs.srv import SetLocalPosition, SetLocalPositionRequest


class PosCmdToLocalGoalBridge(object):
    def __init__(self):
        # match your settings.json vehicle key
        self.vehicle_name = rospy.get_param("~vehicle_name", "drone_1")

        self.srv_name = "/airsim_node/local_position_goal"
        rospy.loginfo("Waiting for service %s", self.srv_name)
        rospy.wait_for_service(self.srv_name)
        self.set_local_goal = rospy.ServiceProxy(self.srv_name, SetLocalPosition)
        rospy.loginfo("Connected to %s", self.srv_name)

        self.last_goal = None

        self.sub = rospy.Subscriber(
            "/planning/pos_cmd",
            PositionCommand,
            self.pos_cmd_cb,
            queue_size=1,
        )

    def _same_goal(self, a, b, eps=1e-3):
        if a is None or b is None:
            return False
        if abs(a.position.x - b.position.x) > eps:
            return False
        if abs(a.position.y - b.position.y) > eps:
            return False
        if abs(a.position.z - b.position.z) > eps:
            return False
        # if PositionCommand has yaw field, use it; otherwise ignore
        if hasattr(a, "yaw") and hasattr(b, "yaw"):
            if abs(a.yaw - b.yaw) > eps:
                return False
        return True

    def pos_cmd_cb(self, msg: PositionCommand):
        # avoid hammering the service with identical goals
        if self._same_goal(msg, self.last_goal):
            return
        self.last_goal = msg

        req = SetLocalPositionRequest()

        # Your controller works in ENU (z up).
        # PD controller expects NED (z down). Convert:
        req.x = msg.position.x
        req.y = msg.position.y
        req.z = -msg.position.z      # ENU -> NED

        # If PositionCommand has yaw in radians, convert to degrees.
        yaw_rad = getattr(msg, "yaw", 0.0)
        req.yaw = math.degrees(yaw_rad)

        # THIS WAS MISSING: tell PD controller which vehicle to move
        req.vehicle_name = self.vehicle_name

        try:
            resp = self.set_local_goal(req)
            if not resp.success:
                rospy.logwarn_throttle(
                    1.0,
                    "SetLocalPosition returned success=False: %s",
                    getattr(resp, "message", ""),
                )
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(
                1.0,
                "SetLocalPosition call failed: %s",
                str(e),
            )


if __name__ == "__main__":
    rospy.init_node("pos_cmd_to_local_goal")
    PosCmdToLocalGoalBridge()
    rospy.spin()
