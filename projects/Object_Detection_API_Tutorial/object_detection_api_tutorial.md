##TensorFlow Object Detection API Tutorial（Win10）

###一、项目结构


###二、安装及测试   
####1.1 TensorFlow Models 下载   
注意：为了确保与 TensorFlow 版本兼容（笔者使用1.15.0），建议使用 models v1.13.0 版本，它是 v1 系列的中最后一个版本。在 git bash 中可通过以下代码下载：   


    git clone -b r1.13.0 https://github.com/tensorflow/models.git




####1.2 COCO API 安装



####1.3 Protobuf 安装   


####1.4 TensorFlow Object Detection API 编译安装


####1.5 环境变量设置


###三、数据标注（以 labelme 标注软件为例）   



###四、数据处理（以 labelme 生成的 xml 格式标注文件为例）   
####3.1 xml标注文件解析   


####3.2 标注信息存储至csv文件   


####3.3 生成TFrecord   


###五、模型训练   
####4.1 模型选择和下载   


####4.2 训练参数配置   


####4.3 模型导出



###六、模型推理或应用   
####5.1单张图片目标检测   


####5.2调用摄像头检测   




###七、参考资料
[Installing the Tensorflow Object Detection API](https://gilberttanner.com/blog/installing-the-tensorflow-object-detection-api)    
