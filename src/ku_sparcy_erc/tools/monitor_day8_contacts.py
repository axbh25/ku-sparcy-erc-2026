#!/usr/bin/env python3
"""Read-only, rate-limited bin contact monitor; not a release permission source."""
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from ros_gz_interfaces.msg import Contacts
from ku_sparcy_erc.day8_contacts import names_of,is_bin,is_table,is_robot


def main():
    rclpy.init();n=Node('day8_read_only_contact_monitor')
    counts=dict(bin_table=0,robot_bin=0,other_object_bin=0)
    state={'phase':'not_started','last_pair':None};last=time.monotonic()
    def contact(m):
        for c in m.contacts:
            a,b=names_of(c)
            if not (is_bin(a) or is_bin(b)):continue
            other=b if is_bin(a) else a
            key='bin_table' if is_table(other) else ('robot_bin' if is_robot(other) else 'other_object_bin')
            counts[key]+=1
            if key!='bin_table':state['last_pair']=[a,b]
    def status(m):
        try:state['phase']=json.loads(m.data)['state']
        except (ValueError,KeyError):pass
    n.create_subscription(Contacts,'/bin_contacts',contact,qos_profile_sensor_data)
    n.create_subscription(String,'/ku_sparcy/day8/status',status,10)
    try:
        while rclpy.ok():
            rclpy.spin_once(n,timeout_sec=.10)
            if time.monotonic()-last>=1.0:
                print(json.dumps(dict(counts=counts,**state)),flush=True);last=time.monotonic()
    except KeyboardInterrupt:pass
    finally:
        n.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
