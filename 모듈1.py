import cv2                                  # 웹캠을 제어하기 위해 opencv 사용 
import mediapipe as mp                      # 손가락인식을 위한 mediapipe 모듈사용


mp_drawing = mp.solutions.drawing_utils     # 웹캠영상에서 포인트와 마디 부분을 그려주는 유틸리티 
mp_hands = mp.solutions.hands
 
cap = cv2.VideoCapture(0)                   # opencv를 이용한 웹캠 제어


# #finger-detection 초기화
with mp_hands.Hands(
    max_num_hands=1,                        # 웹캠에서 최대 인식할 수 있는 손의 갯수 ex) 1=> 웹캠상에서 최대 1개의 손만 인식 
    min_detection_confidence=0.5,           # 0.5로 해야 인식률이 가장 좋음(빵형 영상 참고)=> 구체적인 근거 필요
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():                   # 조건문 (카메라가 계속 작동하면)
        success, image = cap.read()         # cap.read() - 한프레임씩 읽어 오는 코드
        if not success:
            continue
                                                                        # opencv : BGR체계사용,  mediapipe = RGB체계사용 
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)     # opencv에서 입력받은 영상을 mediapipe를 사용하기 위해 RGB로 전환해야함
                                                                        # cv2.flip() - 웹캠영상이 화면상에 출력될 때 좌우가 바뀌어서 나타나기 때문에 좌우 반전을 주기 위한 함수
        results = hands.process(image)                                  # hands.process() - mediapipe에서 전처리 및 모델추론을 함께 실행해주는 함수
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                  # 다시 opencv를 사용하여 화면상에 영상을 출력하기 위해 BGR영상으로 전환

        if results.multi_hand_landmarks:                                # 손이 인식되면 조건문 "참" 성립 => 손가락 포인트 인식
            for hand_landmarks in results.multi_hand_landmarks:
                thumb = hand_landmarks.landmark[4]                      # 각 손가락 배열원소에 포인트 좌표가 저장되어 있음
                index = hand_landmarks.landmark[8]
                
             
                # cv2.putText(
                #     image, text='Volume: %d' % volume, org=(10, 30),
                #     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                #     color=255, thickness=2)

                mp_drawing.draw_landmarks(                             # 손가락 포인트와 마디를 그려주는 함수
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
