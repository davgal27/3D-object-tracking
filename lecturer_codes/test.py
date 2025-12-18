# Grab_MultipleCameras.cpp
# ============================================================================
# This sample illustrates how to grab and process images from multiple cameras
# using the CInstantCameraArray class. The CInstantCameraArray class represents
# an array of instant camera objects. It provides almost the same interface
# as the instant camera for grabbing.
# The main purpose of the CInstantCameraArray is to simplify waiting for images and
# camera events of multiple cameras in one thread. This is done by providing a single
# RetrieveResult method for all cameras in the array.
# Alternatively, the grabbing can be started using the internal grab loop threads
# of all cameras in the CInstantCameraArray. The grabbed images can then be processed by one or more
# image event handlers. Please note that this is not shown in this example.
# ============================================================================

import os
import cv2

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys

# Number of images to be grabbed.
countOfImagesToGrab = 10000

# Limits the amount of cameras used for grabbing.
# It is important to manage the available bandwidth when grabbing with multiple cameras.
# This applies, for instance, if two GigE cameras are connected to the same network adapter via a switch.
# To manage the bandwidth, the GevSCPD interpacket delay parameter and the GevSCFTD transmission delay
# parameter can be set for each GigE camera device.
# The "Controlling Packet Transmission Timing with the Interpacket and Frame Transmission Delays on Basler GigE Vision Cameras"
# Application Notes (AW000649xx000)
# provide more information about this topic.
# The bandwidth used by a FireWire camera device can be limited by adjusting the packet size.
maxCamerasToUse = 2

# The exit code of the sample application.
exitCode = 0

# Example of a device specific handler for image events.
class SampleImageEventHandler(pylon.ImageEventHandler):
    def OnImageGrabbed(self, camera, grabResult):
        # The chunk data is attached to the grab result and can be accessed anywhere.

        # Native parameter access:
        # When using the device specific grab results the chunk data can be accessed
        # via the members of the grab result data.
        if genicam.IsReadable(grabResult.ChunkTimestamp):
            print("OnImageGrabbed: TimeStamp (Result) accessed via result member: ", grabResult.ChunkTimestamp.Value)

try:
    # Get the transport layer factory.
    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RUNTIME_EXCEPTION("No camera present.")

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

    l = cameras.GetSize()

    # Create and attach all Pylon Devices.
    for i, cam in enumerate(cameras):
        #cam.RegisterImageEventHandler(SampleImageEventHandler(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
        #cam.Open()
        #cam.Attach(tlFactory.CreateDevice(devices[i]))
        #cam.StaticChunkNodeMapPoolSize = cam.MaxNumBuffer.GetValue()

        #if genicam.IsWritable(cam.ChunkModeActive):
        #    cam.ChunkModeActive = True
        #else:
        #    raise pylon.RUNTIME_EXCEPTION("The camera doesn't support chunk features")
        #cam.ChunkSelector = "Timestamp"
        #cam.ChunkEnable = True

        # Enable CRC checksum chunks.
        #cam.ChunkSelector = "PayloadCRC16"
        #cam.ChunkEnable = True

        # Print the model name of the camera.
        print("Using device ", cam.GetDeviceInfo().GetModelName())

    # Starts grabbing for all cameras starting with index 0. The grabbing
    # is started for one camera after the other. That's why the images of all
    # cameras are not taken at the same time.
    # However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
    # According to their default configuration, the cameras are
    # set up for free-running continuous acquisition.
    cameras.StartGrabbing()

    # Grab c_countOfImagesToGrab from the cameras.
    for i in range(countOfImagesToGrab):
        if not cameras.IsGrabbing():
            break

        grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # When the cameras in the array are created the camera context value
        # is set to the index of the camera in the array.
        # The camera context is a user settable value.
        # This value is attached to each grab result and can be used
        # to determine the camera that produced the grab result.
        cameraContextValue = grabResult.GetCameraContext()

        print("GRAB ", cameraContextValue)#, grabResult.ChunkTimestamp.Value)

        #img = grabResult.GetArray()
        #cv2.imshow('{}'.format(cameraContextValue), img)
        #print("Gray value of first pixel: ", img[0, 0])
        #if cameraContextValue == 0:
        #    cv2.waitKey(1)

except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", e.GetDescription())
    exitCode = 1

# Comment the following two lines to disable waiting on exit.
sys.exit(exitCode)