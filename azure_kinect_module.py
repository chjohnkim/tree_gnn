import open3d as o3d
import numpy as np
import argparse

class AzureKinectModule:
    def __init__(self, config, device, align_depth_to_color):
        self.flag_exit = False
        self.align_depth_to_color = align_depth_to_color

        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(device):
            raise RuntimeError('Failed to connect to sensor')
        # The intrinisic parameters are obtained from get_camera_info.py
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(1024, 1024, 504.21755981, 504.19558716, 502.46243286, 503.10699463)

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def run_viewer(self):
        glfw_key_escape = 256
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.create_window('viewer', 1920, 540)
        print("Sensor initialized. Press [ESC] to exit.")

        vis_geometry_added = False
        while not self.flag_exit:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            if rgbd is None:
                continue
            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            vis.update_geometry(rgbd)
            vis.poll_events()
            vis.update_renderer()

    def get_frame(self):
        while True:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            if rgbd is None:
                continue
            color = np.asarray(rgbd.color).astype(np.uint8)
            depth = np.asarray(rgbd.depth).astype(np.float32) / 1000.0
            print('Frame caputred with color shape: ', color.shape, ' depth shape: ', depth.shape)
            return color, depth


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect general module.')
    parser.add_argument('--config', type=str, help='input json kinect config')
    parser.add_argument('--list',
                        action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('-a',
                        '--align_depth_to_color',
                        action='store_true',
                        help='enable align depth image to color')
    args = parser.parse_args()
    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0
    cam = AzureKinect(config, device, args.align_depth_to_color)
    cam.run_viewer()