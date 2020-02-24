# Kid's Math Game
家里有两个幼儿园的小朋友，想着给她们搞一个做算数的小游戏。几个游戏平台Pad之类的有些挺好的，但是有些交互太复杂，交互好的又比较贵~
那么...自己动手做一个吧

### 游戏：
* 算数，20以内加减法 （幼儿园）
* 消除类游戏，吸引注意力，生产一堆数字球，算题消除泡泡
* 交互，键盘输入，主要用于调式
* 交互，手写数字，便于小朋友认知，数字纸片用于算题

### 应用技术：
* pygame 游戏引擎，用于构建游戏主体逻辑
* pymunk 物理引擎，数字球运动计算
* opencv 摄像头获取图片用于识别
* tensorflow keras 的手续识别模型，用于数字识别

文件 | 说明
---|---
digit_detect.ipynb | 数字识别的TF模型训练，用的是Keras的mnist库
digits_detect_camera.ipynb | Opencv 从摄像头的图像里识别数字区域，切块标准化后给TF模型进行识别
spawns.py | pymunk的样例，数字求运动
camera.py | 摄像头数字识别
game_camera.py | 游戏程序

![image](https://github.com/chenxinma/kids_math_game/raw/master/doc/screen_01.png)
