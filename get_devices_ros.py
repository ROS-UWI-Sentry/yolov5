#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def talker(number_of_camera):
    pub = rospy.Publisher('/status', String, queue_size=500, latch=True)
    rospy.init_node('talkerstatus')
    pub.publish(number_of_camera)
    rospy.loginfo("number of webcams detected: " + number_of_camera)
 

if __name__ == '__main__':
    with open("streams.txt") as f:
        temp=f.read().splitlines()
    print(temp)
    


    for i in range(len(temp)):
        tempVal=temp[i]
        slashPosition=tempVal.find('/')
        tempVal3=tempVal[0:slashPosition]
        tempVal2=tempVal.replace(tempVal3,'')
        temp[i]=tempVal2
        print(temp[i])
        

    with open("streams.txt",'w') as f:
        for i in range(len(temp)):
            f.write(temp[i]+"\n")
        length=len(temp)
    
    try:
        talker(str(length))
    except rospy.ROSInterruptException:
        pass