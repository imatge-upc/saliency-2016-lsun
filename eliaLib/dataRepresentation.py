import cv2
import numpy as np
from enum import Enum
import scipy.io

class InputType( Enum ):
	image = 0
	imageGrayscale = 1
	saliencyMapMatlab = 2
	empty = 100

class LoadState( Enum ):
	unloaded = 0
	loaded = 1
	loadedCompressed = 2
	error = 100
		
###############################################################################################

class ImageContainer:
	
	def __init__( self, filePath, imageType, state = LoadState.unloaded ):
	
		self.filePath = filePath
		self.state = state
		self.imageType = imageType
				
		if self.state == LoadState.unloaded:
			self.data = None
		elif self.state == LoadState.loaded:
			self.load()
		elif self.state == LoadState.loadedCompressed:
			self.loadCompressed()
		else:
			raise Exception( 'Unknown state when loading image' )
		
	def load( self ):
	
		if self.imageType == InputType.image:
			self.data = cv2.cvtColor( cv2.imread( self.filePath, cv2.CV_LOAD_IMAGE_COLOR ), cv2.cv.CV_BGR2RGB )
			self.state = LoadState.loaded
		if self.imageType == InputType.imageGrayscale:
			self.data = cv2.cvtColor( cv2.imread( self.filePath, cv2.CV_LOAD_IMAGE_COLOR ), cv2.cv.CV_BGR2GRAY )
			self.state = LoadState.loaded
		elif self.imageType == InputType.saliencyMapMatlab:
			self.data = ( scipy.io.loadmat( self.filePath )['I'] * 255 ).astype( np.uint8 )
			self.state = LoadState.loaded
		elif self.imageType == InputType.empty:
			self.data = None
			
	def loadCompressed( self ):

		if self.imageType == InputType.image:
			with open( self.filePath, 'rb' ) as f:
				data = f.read()
			self.data = np.fromstring( data, np.uint8 )
			self.state = LoadState.loadedCompressed
		elif self.imageType == InputType.saliencyMapMatlab:
			self.data = ( scipy.io.loadmat( self.filePath )['I'] * 255 ).astype( np.uint8 )
			self.data = np.squeeze(cv2.imencode('.png', self.data )[1])
			self.state = LoadState.loadedCompressed
			raise Exception('Saliency maps do no have compressed handlind method enabled')
		elif self.imageType == InputType.empty:
			self.state = LoadState.error
			raise Exception('Empty images do no have compressed handlind method enabled')
		
	def getImage( self ):
		
		if self.imageType == InputType.image:
			if self.state == LoadState.unloaded:
				return cv2.cvtColor(cv2.imread( self.filePath, cv2.CV_LOAD_IMAGE_COLOR ), cv2.cv.CV_BGR2RGB )
			elif self.state == LoadState.loaded:
				return self.data
			elif self.state == LoadState.loadedCompressed:
				return cv2.cvtColor(cv2.imdecode(self.data ,cv2.CV_LOAD_IMAGE_COLOR), cv2.cv.CV_BGR2RGB )
		elif self.imageType == InputType.imageGrayscale:
			if self.state == LoadState.unloaded:
				return cv2.cvtColor(cv2.imread( self.filePath, cv2.CV_LOAD_IMAGE_COLOR ), cv2.cv.CV_BGR2GRAY )
			elif self.state == LoadState.loaded:
				return self.data
			elif self.state == LoadState.loadedCompressed:
				return cv2.cvtColor(cv2.imdecode(self.data ,cv2.CV_LOAD_IMAGE_COLOR), cv2.cv.CV_BGR2GRAY )
		elif self.imageType == InputType.saliencyMapMatlab:
			if self.state == LoadState.unloaded:
				return ( scipy.io.loadmat( self.filePath )['I'] * 255 ).astype( np.uint8 )
			elif self.state == LoadState.loaded:
				return self.data
			elif self.state == LoadState.loadedCompressed:
				return cv2.imdecode( self.data, cv2.CV_LOAD_IMAGE_GRAYSCALE )
		elif self.imageType == InputType.empty:
			return None
		
###############################################################################################
			
class Target():

	def __init__( self, imagePath, saliencyPath, 
				  imageState = LoadState.unloaded, imageType = InputType.image, 
				  saliencyState = LoadState.unloaded, saliencyType = InputType.saliencyMapMatlab ):
		
		self.image = ImageContainer( imagePath, imageType, imageState )
		self.saliency = ImageContainer( saliencyPath, saliencyType, saliencyState )
		
###############################################################################################
		
class DataAugmentation():

	def __init__( self, maxAngle, maxScale, maxShift, maxSkew, flip, outputWidth, outputHeight ):
	
		self.maxAngle = maxAngle
		self.maxScale = maxScale
		self.maxSkew = maxSkew
		self.doFlip = flip
		self.outputWidth = outputWidth
		self.outputHeight = outputHeight
		
	def getFlip( self ):
		if self.doFlip:
			return rnd.choice([True, False]) 
		else:
			return False;
		
	def getAngle( self ):
		return np.deg2rad( self.maxAngle * rnd.uniform( -1.0, 1.0 ) )
		
	def getShiftX( self ):
		return self.outputWidth * self.maxShift * rnd.uniform( -1.0, 1.0 )
		
	def getShiftY( self ):
		return self.outputHeight * self.maxShift * rnd.uniform( -1.0, 1.0 )
		
	def getSkew( self ):
		return self.maxSkew * rnd.uniform( -1.0, 1.0 )
		
	def getScaleX( self, inputWidth ):
		scaleX = rnd.uniform( 1.0, self.maxScale ) 
		
		if rnd.choice( [ True, False ] ): 
			scaleX = 1.0 / scaleX 
				
		return scaleX #* ( inputWidth - 1 ) * 1.0 / ( self.outputWidth - 1 )
		
	def getScaleY( self, inputHeight ):
		scaleY = rnd.uniform( 1.0, self.maxScale ) 
		
		if rnd.choice( [ True, False ] ): 
			scaleY = 1.0 / scaleY 
		
		return scaleY #* ( inputHeight - 1 ) * 1.0 / ( self.outputHeight - 1 )
		
	def generateMapping( self, targetRectangle, angle, scaleX, scaleY, skewX, skewY, shiftX, shiftY ):
    
		centerX, centerY = targetRectangle.center().as_tuple()
		targetWidth, targetHeight = targetRectangle.width(), targetRectangle.height()
		
		cosA = np.cos(angle)
		sinA = np.sin(angle)
		
		halfOutputWidth = targetWidth / 2.
		halfOutputHeight = targetHeight / 2.
			
		ctr_in = np.array([centerX, centerY])
		ctr_out = np.array([halfOutputWidth,halfOutputHeight ])
		
		transform = np.array([[cosA, -sinA], [sinA, cosA]])
		transform = transform.dot(np.array([[1.0, skewY], [0.0, 1.0]]))
		transform = transform.dot(np.array([[1.0, 0.0], [skewX, 1.0]]))
		transform = transform.dot(np.diag([scaleX, scaleY]))
		
		offset = ctr_in-ctr_out.dot(transform.T)
		
		mapping = np.vstack((transform.T,offset.T)).T
		
		return mapping
		
	def generateImage( self, currTarget ):
		
		ctr_in = np.array((currTarget.bbox.center().y, currTarget.bbox.center().x))
		
		currShiftX = self.getShiftX()
		currShiftY = self.getShiftY()
		
		ctr_out = np.array((self.outputHeight/2.0+currShiftY, self.outputWidth/2.0+currShiftX))
		out_shape = (self.outputHeight, self.outputWidth)
		
		# rotation angle 
		currAngle = self.getAngle()
		
		#skew 
		currSkewX = self.getSkew()
		currSkewY = self.getSkew()
		
		# scale 
		
		currScaleY = self.getScaleY( currTarget.bbox.height() )
		currScaleX = self.getScaleX( currTarget.bbox.width() )
		
		mapping = self.generateMapping( currTarget.bbox,\
									    currAngle, currScaleY, currScaleX,
									    currSkewX, currSkewY,
									    currShiftX, currShiftY )
		
		#outImg = cv2.warpAffine( currTarget.image.getImage(), mapping, ( self.outputWidth, self.outputHeight ), flags = cv2.WARP_INVERSE_MAP )
		outImg = cv2.warpAffine( currTarget.image.getImage(), mapping, ( currTarget.bbox.width(), currTarget.bbox.height() ), flags = cv2.WARP_INVERSE_MAP )
		outImg = cv2.resize(outImg, (self.outputWidth, self.outputHeight) )
		if self.getFlip(): 
			outImg = np.fliplr( outImg )
			
		return outImg