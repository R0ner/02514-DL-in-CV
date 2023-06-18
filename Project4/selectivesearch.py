import cv2

class SelectiveSearch():
    def __init__(self, mode='f', nkeep=100):
        self.mode = mode
        self.nkeep = nkeep
        
           
    def __call__(self, im):
        """
        Args:
            im: np.ndarray image with shape (H x W x C) and dtype np.uint8 (0-255)
        """
        self.gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.gs.setBaseImage(im)

        self.set_mode()

        boxes = self.gs.process()

        return boxes[:self.nkeep]
    
    def set_mode(self):
        if (self.mode == 's'):
            self.gs.switchToSingleStrategy()
        elif (self.mode == 'f'):
            self.gs.switchToSelectiveSearchFast()
        elif (self.mode == 'q'):
            self.gs.switchToSelectiveSearchQuality()
        else:
            raise