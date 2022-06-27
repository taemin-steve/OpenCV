import numpy as np
import cv2 as cv
import math 

FLANN_INDEX_LSH    = 6
def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2):

    flann_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2

    matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k = 2) #2

    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.79:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) >= 4:
        keyPoints1 = np.float32([keyPoints1[i] for (_, i) in matches])
        keyPoints2 = np.float32([keyPoints2[i] for (i, _) in matches])
        H, status = cv.findHomography(keyPoints1, keyPoints2, cv.RANSAC , 5.0)
    else:
        H, status = None, None
        
    return matches, H, status


   
def drawMatches(image1, image2, keyPoints1, keyPoints2, matches, status): #호모 그래피를 추가해주는 함수 

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    img_matching_result = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")

    img_matching_result[0:h2, 0:w2] = image2
    img_matching_result[0:h1, w2:] = image1

    for ((trainIdx, queryIdx), s) in zip(matches, status):

        if s == 1:
            keyPoint2 = (int(keyPoints2[trainIdx][0]), int(keyPoints2[trainIdx][1]))
            keyPoint1 = (int(keyPoints1[queryIdx][0]) + w2, int(keyPoints1[queryIdx][1]))
            cv.line(img_matching_result, keyPoint1, keyPoint2, (0, 255, 0), 1)
    return img_matching_result


def main():


    img1 = cv.imread('./3.jpg') 
    img2 = cv.imread('./2.jpg') 
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    detector = cv.BRISK_create()
    keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None) # 키포인트와 디스크립터를 생성

    keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
    keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])
    
    matches, H, status = matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2) #필요한 호모그래피 메트릭스를 생성해준다. 

    img_matching_result = drawMatches(img1, img2, keyPoints1, keyPoints2, matches, status)
    
    ## 이미지 영역을 벗어나는 부분을 고려해주기 위해서 호모그래피를 이용하여 원래 영상의 이동된 좌표를 계산해주는 부분
    point = [0,0,1]
    A = np.array(point).transpose()
    point = [img1.shape[1],0,1]
    B = np.array(point).transpose()
    point = [0,img1.shape[0],1]
    C = np.array(point).transpose()
    point = [img1.shape[1],img1.shape[0],1]
    D = np.array(point).transpose()

    conerX = []
    conerY = []
    
    matA = H.copy()
    matB = H.copy()
    matC = H.copy()
    matD = H.copy()
    
    AA =np.matmul(matA,A)
    BB =np.matmul(matB,B)
    CC =np.matmul(matC,C)
    DD =np.matmul(matD,D)
    
    conerX.append(AA[0]/AA[2])
    conerX.append(BB[0]/BB[2])
    conerX.append(CC[0]/CC[2])
    conerX.append(DD[0]/DD[2])
    
    conerY.append(AA[1]/AA[2])
    conerY.append(BB[1]/BB[2])
    conerY.append(CC[1]/CC[2])
    conerY.append(DD[1]/DD[2])
    conerY.append(img1.shape[0])
    
    middleBoundary = [[0,0], [img1.shape[1],0],[0,img1.shape[0]],[img1.shape[1],img1.shape[0]]]
    

    ## 실제로 이미지 영역이 벗어난다면 진행
    if min(conerY) < 0:
        dx = 0
        dy = math.ceil( min(conerY))
        mtrx = np.float32([[1, 0, dx],
                   [0, 1, -dy]])
        
        img1 = cv.warpAffine(img1, mtrx, (img1.shape[1]+dx, img1.shape[0]-dy) )
        img2 = cv.warpAffine(img2, mtrx, (img2.shape[1]+dx, img2.shape[0]-dy) )
        middleBoundary[0][1] = middleBoundary[0][1] - dy
        middleBoundary[1][1] = middleBoundary[1][1] - dy
        middleBoundary[2][1] = middleBoundary[2][1] - dy
        middleBoundary[3][1] = middleBoundary[3][1] - dy
        
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        
        grayImgFor = cv.imread("./gray.jpg")
        if(type(grayImgFor) == type(None)):
            pass
        else:
            newGray = cv.resize(grayImgFor, (img1.shape[1] , img1.shape[0]), interpolation=cv.INTER_AREA) 
        grayH = cv.warpAffine(newGray, mtrx, (img1.shape[1]+dx, img1.shape[0]-dy) ) 
        
        detector = cv.BRISK_create()
        keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
        keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)

        keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
        keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])
        
        matches, H, status = matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2)
        
        
        point = [0,0,1]
        A = np.array(point).transpose()
        point = [img1.shape[1],0,1]
        B = np.array(point).transpose()
        point = [0,img1.shape[0],1]
        C = np.array(point).transpose()
        point = [img1.shape[1],img1.shape[0],1]
        D = np.array(point).transpose()

        conerX = []
        conerY = []

        matA = H.copy()
        matB = H.copy()
        matC = H.copy()
        matD = H.copy()

        AA =np.matmul(matA,A)
        BB =np.matmul(matB,B)
        CC =np.matmul(matC,C)
        DD =np.matmul(matD,D)

        conerX.append(AA[0]/AA[2])
        conerX.append(BB[0]/BB[2])
        conerX.append(CC[0]/CC[2])
        conerX.append(DD[0]/DD[2])

        conerY.append(AA[1]/AA[2])
        conerY.append(BB[1]/BB[2])
        conerY.append(CC[1]/CC[2])
        conerY.append(DD[1]/DD[2])
        conerY.append(img2.shape[0])

        result = cv.warpPerspective(img1, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
        grayH = cv.warpPerspective(grayH, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
    else:
        if(type(grayImgFor) == type(None)):
            pass
        else:
            newGray = cv.resize(grayImgFor, (img1.shape[1], img1.shape[0]), interpolation=cv.INTER_AREA) 
        grayH = cv.warpAffine(newGray, mtrx, (img1.shape[1]+dx, img1.shape[0]-dy) ) #/ 여기 한번 봐주고
        
        grayH = cv.warpPerspective(grayH, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
        result = cv.warpPerspective(img1, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
   
    mtrx = np.float32([[1, 0, 0],
                   [0, 1, 0]])
    img2 = cv.warpAffine(img2, mtrx, (result.shape[1], result.shape[0]) )
    
    src1 = np.array(img2)
    src2 = np.array(result)
    mask1 = np.array(grayH)
    print(src1.shape, mask1.shape)
    
    ## 마스크필터를 적용하여 이미지 블랜딩을 자연스럽게 진행
    mask1 = mask1 / 255

    dst = src2 * mask1 + src1 * (1 - mask1) 
    resultN = dst
    resultN = np.reshape(resultN, (img2.shape[0], img2.shape[1],3))
    
    cv.imwrite('./resultN.png',resultN)
    rww = cv.imread("./resultN.png")
    
    cv.imshow('1', rww)
    
    resultN = rww
# /////////////////////
 #아래서 부터는 위와 동일하게 나머지 이미지에 대해서 진행한다.
    img1 = cv.imread('1.jpg') 
    img2 = resultN
    
    img1 = cv.rotate(img1, cv.ROTATE_180)
    img2 = cv.rotate(img2, cv.ROTATE_180)

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    
 
    detector = cv.BRISK_create()
    keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)


    
    keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
    keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])
    


    matches, H, status = matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2)
    img_matching_result = drawMatches(img1, img2, keyPoints1, keyPoints2, matches, status)
    point = [0,0,1]
    A = np.array(point).transpose()
    point = [img1.shape[1],0,1]
    B = np.array(point).transpose()
    point = [0,img1.shape[0],1]
    C = np.array(point).transpose()
    point = [img1.shape[1],img1.shape[0],1]
    D = np.array(point).transpose()

    conerX = []
    conerY = []
    
    matA = H.copy()
    matB = H.copy()
    matC = H.copy()
    matD = H.copy()
    
    AA =np.matmul(matA,A)
    BB =np.matmul(matB,B)
    CC =np.matmul(matC,C)
    DD =np.matmul(matD,D)
    
    conerX.append(AA[0]/AA[2])
    conerX.append(BB[0]/BB[2])
    conerX.append(CC[0]/CC[2])
    conerX.append(DD[0]/DD[2])
    
    conerY.append(AA[1]/AA[2])
    conerY.append(BB[1]/BB[2])
    conerY.append(CC[1]/CC[2])
    conerY.append(DD[1]/DD[2])
    conerY.append(img1.shape[0])

    
    if min(conerY) < 0:
        dx = 0
        dy =math.floor( min(conerY))
        mtrx = np.float32([[1, 0, dx],
                   [0, 1, -dy]])
        img1 = cv.warpAffine(img1, mtrx, (img1.shape[1]+dx, img1.shape[0]-dy) )
        img2 = cv.warpAffine(img2, mtrx, (img2.shape[1]+dx, img2.shape[0]-dy) )
        
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        
        if(type(grayImgFor) == type(None)):
            pass
        else:
            newGray = cv.resize(grayImgFor, (img1.shape[1] , img1.shape[0]), interpolation=cv.INTER_AREA) 
        grayH = cv.warpAffine(newGray, mtrx, (img1.shape[1]+dx, img1.shape[0]-dy) ) #/ 여기 한번 봐주고
        
        detector = cv.BRISK_create()
        keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
        keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)

        keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
        keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])
        
        matches, H, status = matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2)

        img_matching_result = drawMatches(img1, img2, keyPoints1, keyPoints2, matches, status)

        point = [0,0,1]
        A = np.array(point).transpose()
        point = [img1.shape[1],0,1]
        B = np.array(point).transpose()
        point = [0,img1.shape[0],1]
        C = np.array(point).transpose()
        point = [img1.shape[1],img1.shape[0],1]
        D = np.array(point).transpose()

        conerX = []
        conerY = []

        matA = H.copy()
        matB = H.copy()
        matC = H.copy()
        matD = H.copy()

        AA =np.matmul(matA,A)
        BB =np.matmul(matB,B)
        CC =np.matmul(matC,C)
        DD =np.matmul(matD,D)

        conerX.append(AA[0]/AA[2])
        conerX.append(BB[0]/BB[2])
        conerX.append(CC[0]/CC[2])
        conerX.append(DD[0]/DD[2])

        conerY.append(AA[1]/AA[2])
        conerY.append(BB[1]/BB[2])
        conerY.append(CC[1]/CC[2])
        conerY.append(DD[1]/DD[2])
        conerY.append(img2.shape[0])
        
        grayH = cv.warpPerspective(grayH, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
        result2 = cv.warpPerspective(img1, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
    else:
        if(type(grayImgFor) == type(None)):
            pass
        else:
            newGray = cv.resize(grayImgFor, (img1.shape[1], img1.shape[0]), interpolation=cv.INTER_AREA) 
        grayH = cv.warpAffine(newGray, mtrx, (img1.shape[1]+dx, img1.shape[0]-dy) )
        
        grayH = cv.warpPerspective(grayH, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
        result2 = cv.warpPerspective(img1, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
   
    mtrx = np.float32([[1, 0, 0],
                   [0, 1, 0]])
    img2 = cv.warpAffine(img2, mtrx, (result2.shape[1], result2.shape[0]) )
    
    src1 = np.array(img2)
    src2 = np.array(result2)
    mask1 = np.array(grayH)
    print(src1.shape, src2.shape, mask1.shape)
    
    mask1 = mask1 / 255

    dst = src2 * mask1 + src1 * (1 - mask1) 
    resultN = dst
    resultN = np.reshape(resultN, (img2.shape[0], img2.shape[1],3))
    
    cv.imwrite('./resultN.png',resultN)
    rww = cv.imread("./resultN.png")
    


    
    result2 = cv.rotate(rww, cv.ROTATE_180)
    
    cv.imshow('3', result2)
    cv.waitKey()


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()