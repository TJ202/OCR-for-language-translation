# OCR-for-language-translation
We intend to translate text in images from one language to the other by developing an OCR
model. For doing this, scanning, pre-processing, segmentation, feature extraction from the
images and training of the model is necessary. We used the EAST text detection model to draw
bounding boxes around text which is then fed to Tesseract (an OCR engine) which recognises
text. The performance can be assessed on a character or on a word level by taking the output of
an image and comparing it with the original version of the text.
