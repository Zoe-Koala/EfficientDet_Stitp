#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from efficientdet import EfficientDet
from PIL import Image
import numpy as np
import cv2
import time

from efficientdet import position_st

efficientdet = EfficientDet()
# 调用摄像头
# capture=cv2.VideoCapture(0)

# 检测本地视频
capture=cv2.VideoCapture("2.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('./video_record/camera_record_1.avi', fourcc,20.0, size)

fps = 0.0

start = 0
count_st = 0
time_judge = [0]
time_count = 0

frame_no_head = []
count_no_head = 0
time_no_head = []

frame_head_danger = []
count_head_danger = 0
time_head_danger = []

frame_no_hand = []
count_no_hand = 0
time_no_hand = []

frame_one_hand = []
count_one_hand = 0
time_one_hand = []

frame_hand_1_danger = []
count_hand_1_danger = 0
time_hand_1_danger = []

frame_hand_2_danger = []
count_hand_2_danger = 0
time_hand_2_danger = []

list_head = list(range(0,20))
list_hand = list(range(0,20))
distance_head = list(range(0,20))
distance_hand_1 = list(range(0,20))
distance_hand_2 = list(range(0,20))

while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = np.array(efficientdet.detect_image(frame))

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2

    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # cv2.imshow("video",frame)

    start = start + 1

    # print(position_st)
    # print(distance)

    if start >= 20:
        for position_st_name,position_st_info in position_st.items():
            head_name = position_st_info['head']
            list_head[count_st % 20] = head_name
            hand_name = position_st_info['hand']
            list_hand[count_st % 20] = hand_name
        
        count_st = count_st + 1
            
            # print(head_name)
            # print(hand_name)

        # print(list_head)
        # print(list_hand)
        if count_st >= 20:
            # print(list_head)
            if (list_head[0] and list_head[1] and list_head[2]):
                x_head = (list_head[0][0] + list_head[1][0] + list_head[2][0]) / 3
                y_head = (list_head[0][1] + list_head[1][1] + list_head[2][1]) / 3
                if list_head[count_st % 20]:
                    distance_head[count_st % 20] = ((list_head[count_st % 20][0] - x_head)**2 + (list_head[count_st % 20][1] - y_head)**2)**0.5
                else:
                    distance_head[count_st % 20] = 40    # 待修改

                if distance_head[count_st % 20] >= 30:   # 待修改
                    # print("head----->danger!!!!!!!!!!")
                    frame = cv2.putText(frame, "head in danger", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # out.write(frame)
                    frame_head_danger.append(frame)
                    time_head_danger.append(t1)
                    count_head_danger = count_head_danger + 1
                    if count_head_danger < 20 and time_head_danger[count_head_danger-1] - time_head_danger[0]>3:
                        frame_head_danger = []
                        count_head_danger = 0
                        time_head_danger = []

                    if count_head_danger > 20 and time_head_danger[count_head_danger-1] - time_head_danger[0]>3 and time_head_danger[0] > time_judge[time_count]:
                        for x in range(count_head_danger-1):
                            out.write(frame_head_danger[x])
                        frame_head_danger = []
                        count_head_danger = 0
                        time_head_danger = []
                        time_judge.append(t1)
                        time_count = time_count + 1


                else:
                    # print("head---->safe~~~~~~~~~~")
                    frame = cv2.putText(frame, "head in safe", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # print("no--->head!!!!")
                frame = cv2.putText(frame, "no head", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frame_no_head.append(frame)
                time_no_head.append(t1)
                count_no_head = count_no_head + 1
                if count_no_head < 20 and time_no_head[count_no_head-1] - time_no_head[0]>3:
                    frame_no_head = []
                    count_no_head = 0
                    time_no_head = []

                if count_no_head > 20 and time_no_head[count_no_head-1] - time_no_head[0]>3 and time_no_head[0] > time_judge[time_count]:
                    for x in range(count_no_head-1):
                        out.write(frame_no_head[x])
                    frame_no_head = []
                    count_no_head = 0
                    time_no_head = []
                    time_judge.append(t1)
                    time_count = time_count + 1


            if list_hand[0] and list_hand[1] and list_hand[2]:
                if len(list_hand[0])==len(list_hand[1])==len(list_hand[2])==2:
                    # print("两只手都在")
                    x_1_hand=(list_hand[0][0][0]+list_hand[1][0][0]+list_hand[2][0][0])/3
                    y_1_hand=(list_hand[0][0][1]+list_hand[1][0][1]+list_hand[2][0][1])/3
                    x_2_hand=(list_hand[0][1][0]+list_hand[1][1][0]+list_hand[2][1][0])/3
                    y_2_hand=(list_hand[0][1][1]+list_hand[1][1][1]+list_hand[2][1][1])/3
                    ## print(x_1_hand,y_1_hand,x_2_hand,y_2_hand)
                    if len(list_hand[count_st % 20])==2:
                        distance_hand_1[count_st%20]=((list_hand[count_st%20][0][0]-x_1_hand)**2+(list_hand[count_st%20][0][1]-y_1_hand)**2)**0.5
                        distance_hand_2[count_st%20]=((list_hand[count_st%20][1][0]-x_2_hand)**2+(list_hand[count_st%20][1][1]-y_2_hand)**2)**0.5
                        # print(distance_hand_1,distance_hand_2)
                    else:
                        distance_hand_1[count_st%20]=distance_hand_2[count_st%20]=210
                    if distance_hand_1[count_st % 20]>=70:
                        # print("hand_1---->danger!!!!!!")
                        frame = cv2.putText(frame, "hand_1 in danger", (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        frame_hand_1_danger.append(frame)
                        time_hand_1_danger.append(t1)
                        count_hand_1_danger = count_hand_1_danger + 1

                        if count_hand_1_danger < 20 and time_hand_1_danger[count_hand_1_danger-1] - time_hand_1_danger[0]>3:
                            frame_hand_1_danger = []
                            count_hand_1_danger = 0
                            time_hand_1_danger = []    

                        if count_hand_1_danger > 20 and time_hand_1_danger[count_hand_1_danger-1] - time_hand_1_danger[0]>3 and time_hand_1_danger[0] > time_judge[time_count]:
                            for x in range(count_hand_1_danger-1):
                                out.write(frame_hand_1_danger[x])
                            frame_hand_1_danger = []
                            count_hand_1_danger = 0
                            time_hand_1_danger = []
                            time_judge.append(t1)
                            time_count = time_count + 1



                        # out.write(frame)
                    else:
                        # print("hand_1---->safe~~~~~~~~~~")
                        frame = cv2.putText(frame, "hand_1 in safe", (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if distance_hand_2[count_st % 20]>=70:
                        # print("hand_2---->danger!!!!!!")
                        frame = cv2.putText(frame, "hand_2 in danger", (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        frame_hand_2_danger.append(frame)
                        time_hand_2_danger.append(t1)
                        count_hand_2_danger = count_hand_2_danger + 1

                        if count_hand_2_danger < 20 and time_hand_2_danger[count_hand_2_danger-1] - time_hand_2_danger[0]>3:
                            frame_hand_2_danger = []
                            count_hand_2_danger = 0
                            time_hand_2_danger = []

                        if count_hand_2_danger > 20 and time_hand_2_danger[count_hand_2_danger-1] - time_hand_2_danger[0]>3 and time_hand_2_danger[0] > time_judge[time_count]:
                            for x in range(count_hand_2_danger-1):
                                out.write(frame_hand_2_danger[x])
                            frame_hand_2_danger = []
                            count_hand_2_danger = 0
                            time_hand_2_danger = []
                            time_judge.append(t1)
                            time_count = time_count + 1

                        
                        # out.write(frame)
                    else:
                        # print("hand_2---->safe~~~~~~~~~~")
                        frame = cv2.putText(frame, "hand_2 in safe", (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # print("hand is in danger!!")
                    frame = cv2.putText(frame, "just one hand", (325, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    frame_one_hand.append(frame)
                    time_one_hand.append(t1)
                    count_one_hand = count_one_hand + 1

                    if count_one_hand < 20 and time_one_hand[count_one_hand-1] - time_one_hand[0]>3:
                        frame_one_hand = []
                        count_one_hand = 0
                        time_one_hand = []

                    if count_one_hand > 20 and time_one_hand[count_one_hand-1] - time_one_hand[0]>3 and time_one_hand[0] > time_judge[time_count]:
                        for x in range(count_one_hand-1):
                            out.write(frame_one_hand[x])
                        frame_one_hand = []
                        count_one_hand = 0
                        time_one_hand = []
                        time_judge.append(t1)
                        time_count = time_count + 1


                    # out.write(frame)
            else:
                frame = cv2.putText(frame, "no hand", (325, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # out.write(frame)
                frame_no_hand.append(frame)
                time_no_hand.append(t1)
                count_no_hand = count_no_hand + 1

                if count_no_hand < 20 and time_no_hand[count_no_hand-1] - time_no_hand[0]>3:
                    frame_no_hand = []
                    count_no_hand = 0
                    time_no_hand = []

                if count_no_hand > 20 and time_no_hand[count_no_hand-1] - time_no_hand[0]>3 and (time_no_hand[0] > time_judge[time_count]):
                    for x in range(count_no_hand-1):
                        out.write(frame_no_hand[x])
                    frame_no_hand = []
                    count_no_hand = 0
                    time_no_hand = []
                    time_judge.append(t1)
                    time_count = time_count + 1


    # RGBtoBGR满足opencv显示格式
    # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    # fps  = ( fps + (1./(time.time()-t1)) ) / 2
    # print("fps= %.2f"%(fps))
    # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("video",frame)

    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        out.release()
        break




'''

#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from efficientdet import EfficientDet
from PIL import Image
import numpy as np
import cv2
import time

efficientdet = EfficientDet()
# 调用摄像头
capture=cv2.VideoCapture(0)

# 检测本地视频
# capture=cv2.VideoCapture("1.mp4")

fps = 0.0


while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = np.array(efficientdet.detect_image(frame))

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)

    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        break


'''


'''
左右左右左右左右--->两只手一直在的时候，一个隔一个的检测移动距离，可行√
一直一个手--->一个隔一个的分析也可行，只是每隔两帧判断一次，时间也是较短的，可行√

前一段时间一直是一只手，某刻以后一直是两只手--->
------左左左左左右左左右左右：奇数帧检测不出来多的右手，但是偶数帧可以检测出多了一只手
------左左左左左左右左右左右左右：奇数帧可以检测出多的右手且从此往后左变成右

前一段时间一直是两只手，某时刻以后变成只有一只手--->
------左右左右左右左右左右左左左左：奇数帧无法检测出来，但是偶数帧可以检测出少了一只手
------左右左右左右左右左右右右右右：偶数帧无法检测出来，但是奇数帧可以检测出少了一只手
'''

'''
前一种提取方式可能会导致问题
1. 首先是单手双手头部什么的这些数据很乱，不好判断消失的情况
2. 不好定期清除数据，时间长了耗内存
3. 等等等

现在想重新换一种存储方式：
先定好存多少帧的数据，然后覆盖，减少内存消耗
至于怎么判断法


'''