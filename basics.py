import cv2
import face_recognition

# imgElon = face_recognition.load_image_file('Elon-Musk.jpg')
# imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
# imgTest = face_recognition.load_image_file('elon-test.jpg')
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
imgGates = face_recognition.load_image_file('Bill_Gates.jpg')
imgGates = cv2.cvtColor(imgGates, cv2.COLOR_BGR2RGB)
imgGates = cv2.resize(imgGates, (0, 0), fx=0.7, fy=0.7)
#
# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
faceLocGates = face_recognition.face_locations(imgGates)[0]
encodeGates = face_recognition.face_encodings(imgGates)[0]
cv2.rectangle(imgGates,(faceLocGates[3],faceLocGates[0]),(faceLocGates[1],faceLocGates[2]),(255,0,255),2)
#
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
# results = face_recognition.compare_faces([encodeElon],encodeGates)
# print(results)


imgMarwen = face_recognition.load_image_file('marwen_mejri.jpg')
imgMarwen = cv2.cvtColor(imgMarwen, cv2.COLOR_BGR2RGB)
imgMarwen = cv2.resize(imgMarwen, (0, 0), fx=0.4, fy=0.4)

faceLoc = face_recognition.face_locations(imgMarwen)[0]
encodeElon = face_recognition.face_encodings(imgMarwen)[0]
cv2.rectangle(imgMarwen,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

imgTest = face_recognition.load_image_file('marwen_test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
imgTest = cv2.resize(imgTest, (0, 0), fx=0.2, fy=0.2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Elon Musk',imgMarwen)
cv2.imshow('Elon Test',imgTest)
cv2.imshow('Bill Gates',imgGates)

cv2.waitKey(0)