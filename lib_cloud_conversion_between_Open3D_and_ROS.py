#!/usr/bin/env python
# -*- coding: utf-8 -*-

import open3d
import numpy as np
from ctypes import * # convert float to uint32

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = [ 
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="odom"):
    print("Converting Cloud From Open3D To ROS")

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    points=np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255) # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
        cloud_data=np.c_[points, colors]

    return pc2.create_cloud(header, fields, cloud_data)

def convertCloudFromRosToOpen3d(ros_cloud):
    
    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names = field_names))

    # Check empty
    open3d_cloud = open3d.geometry.PointCloud()
    if len(cloud_data)==0:
        print("Converting an empty cloud")
        return None

    # Set open3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb
        
        # Get xyz
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

        # Get rgb
        # Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
            rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

        # combine
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = open3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))

    # return
    return open3d_cloud

# -- Example of usage
if __name__ == "__main__":
    rospy.init_node('test_pc_conversion_between_Open3D_and_ROS', anonymous=True)
    
    import os
    PYTHON_FILE_PATH=os.path.join(os.path.dirname(__file__))+"/"
    # if 1: # test XYZ point cloud format
    #     filename=PYTHON_FILE_PATH+"test_cloud_XYZ_noRGB.pcd"
    # else: # test XYZRGB point cloud format
    #     filename=PYTHON_FILE_PATH+"test_cloud_XYZRGB.pcd"

    # print("Loading Open3D Cloud")
    # open3d_cloud = open3d.io.read_point_cloud(filename)

    # # -- Set publisher
    # topic_name="kinect2/qhd/points"
    # pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
    
    # -- Set subscriber
    global received_ros_cloud
    received_ros_cloud = None
    def callback(ros_cloud):
        global received_ros_cloud
        received_ros_cloud=ros_cloud
        rospy.loginfo("-- Received ROS PointCloud2 message.")
    rospy.Subscriber("/map_compress/full_cloud/3d", PointCloud2, callback)      
    
    # -- Convert open3d_cloud to ros_cloud, and publish. Until the subscribe receives it.
    while received_ros_cloud is None and not rospy.is_shutdown():
        rospy.loginfo("-- Not receiving ROS PointCloud2 message yet ...")

        # if 1: # Use the cloud from file
        #     rospy.loginfo("Converting cloud from Open3d to ROS PointCloud2 ...")
        #     ros_cloud = convertCloudFromOpen3dToRos(open3d_cloud)

        # # publish cloud
        # pub.publish(ros_cloud)
        rospy.sleep(1)
        
    print("Converting To Open3d")
    received_open3d_cloud = convertCloudFromRosToOpen3d(received_ros_cloud)

    # Poisson Surface Reconstruction
    print("Estimating Normals")
    received_open3d_cloud.estimate_normals()
    received_open3d_cloud.orient_normals_consistent_tangent_plane(100)
    # open3d.visualization.draw_geometries([received_open3d_cloud], point_show_normal=True)
    mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(received_open3d_cloud)
    # mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(received_open3d_cloud, 0.03)
    # mesh.compute_vertex_normals()
    # # mesh.paint_uniform_color([0.706, 0.706, 0.706])
    # open3d.visualization.draw_geometries([mesh])
    # open3d.io.write_triangle_mesh(PYTHON_FILE_PATH+"example_mesh.ply", mesh)
    open3d.io.write_point_cloud(PYTHON_FILE_PATH+"example_polygon.ply", received_open3d_cloud)