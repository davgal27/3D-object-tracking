
from pypylon import genicam
from pypylon import pylon
import time
import queue

class ImageEventPrinter(pylon.ImageEventHandler):
    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
        print(countOfSkippedImages, " images have been skipped.")
        print()

    def OnImageGrabbed(self, camera, grabResult):
        print("OnImageGrabbed event for device ", camera.GetDeviceInfo().GetModelName())

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            print("SizeX: ", grabResult.GetWidth())
            print("SizeY: ", grabResult.GetHeight())
            img = grabResult.GetArray()
            print("Gray values of first row: ", img[0])
            print()
        else:
            print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())


class CameraEventPrinter(pylon.CameraEventHandler):
    def OnCameraEvent(self, camera, userProvidedId, node):
        print("OnCameraEvent event for device ", camera.GetDeviceInfo().GetModelName())
        print("User provided ID: ", userProvidedId)
        print("Event data node name: ", node.GetName())
        value = genicam.CValuePtr(node)
        if value.IsValid():
            print("Event node data: ", value.ToString())
        print()


class ConfigurationEventPrinter(pylon.ConfigurationEventHandler):
    def OnAttach(self, camera):
        print("OnAttach event")

    def OnAttached(self, camera):
        print("OnAttached event for device ", camera.GetDeviceInfo().GetModelName())

    def OnOpen(self, camera):
        print("OnOpen event for device ", camera.GetDeviceInfo().GetModelName())

    def OnOpened(self, camera):
        print("OnOpened event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStart(self, camera):
        print("OnGrabStart event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStarted(self, camera):
        print("OnGrabStarted event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStop(self, camera):
        print("OnGrabStop event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStopped(self, camera):
        print("OnGrabStopped event for device ", camera.GetDeviceInfo().GetModelName())

    def OnClose(self, camera):
        print("OnClose event for device ", camera.GetDeviceInfo().GetModelName())

    def OnClosed(self, camera):
        print("OnClosed event for device ", camera.GetDeviceInfo().GetModelName())

    def OnDestroy(self, camera):
        print("OnDestroy event for device ", camera.GetDeviceInfo().GetModelName())

    def OnDestroyed(self, camera):
        print("OnDestroyed event")

    def OnDetach(self, camera):
        print("OnDetach event for device ", camera.GetDeviceInfo().GetModelName())

    def OnDetached(self, camera):
        print("OnDetached event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabError(self, camera, errorMessage):
        print("OnGrabError event for device ", camera.GetDeviceInfo().GetModelName())
        print("Error Message: ", errorMessage)

    def OnCameraDeviceRemoved(self, camera):
        print("OnCameraDeviceRemoved event for device ", camera.GetDeviceInfo().GetModelName())


# Example of an image event handler.
class SampleImageEventHandler(pylon.ImageEventHandler):
    def __init__(self, q, cam_id):
        super().__init__()
        self.grab_times = [0.0]
        self.q = q
        self.cam_id = cam_id

    def OnImageGrabbed(self, camera, grabResult):
        timestamp = time.time()
        cam_id = camera.GetDeviceInfo().GetDeviceGUID()
        self.q.put((cam_id, timestamp, grabResult.GetArray()))


class ImageSource:
    def __init__(self):
        q = queue.Queue(4)
        cameras = []
        handlers = []
        device_info_list = pylon.TlFactory.GetInstance().EnumerateDevices()
        for info in device_info_list:
            cam_id = info.GetDeviceGUID()

            cam = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice(info))
            cam.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(),
                                      pylon.RegistrationMode_ReplaceAll,
                                      pylon.Cleanup_Delete)
            cam.RegisterConfiguration(ConfigurationEventPrinter(),
                                      pylon.RegistrationMode_Append,
                                      pylon.Cleanup_Delete)
            handlers.append(SampleImageEventHandler(q, cam_id))
            cam.RegisterImageEventHandler(handlers[-1],
                                          pylon.RegistrationMode_Append,
                                          pylon.Cleanup_Delete)
            cam.StartGrabbing(pylon.GrabStrategy_OneByOne,
                              pylon.GrabLoop_ProvidedByInstantCamera)
            cameras.append(cam)

        self.q = q
        self.cameras = cameras

    def trigger_cameras(self):
        for cam in self.cameras:
            cam.ExecuteSoftwareTrigger()

    def get_images(self):
        images = [self.q.get() for i in range(len(self.cameras))]
        return images
