from collections import OrderedDict

class ModelInput(object):

	def __init__( self, netName, iInputWidht, iInputHeight, iMeanImage, iScaleFactor ):
				  
		self.netName = netName
		self.inputWidth = iInputWidht
		self.inputHeight = iInputHeight
		
		self.meanImage = iMeanImage
		self.scaleFactor = iScaleFactor
		
		self.outputLayerName = None
		self.inputLayerName = None
			
	def build( self, sharedNet, input_var ):
		return None

		
	def preparaImage( self, img ):
		return None
