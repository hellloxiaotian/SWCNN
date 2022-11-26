## A self-supervised CNN for image watermark by Chunwei Tian, Menghua Zheng, Tiancai Jiao, Zheng Wang, Wangmeng Zuo and Yanning Zhang and it is implemented by Pytorch.

### The usage of data set

The structure of /dataset:
		—— datasets
   				|——train
     						 |——train_data1.zip
      						|——train_data2.zip
   				|——test.zip
   				|——watermark.zip

**datasets/train** contains clean images for training model.

**datasets/test.zip** contains clean images for testing model. 

**datasets/watermark.zip** contains watermark for synthesizing the watermark images.

**/datasets/train/train_data1.zip**, **/datasets/train/train_data2.zip**  and **/datasets/watermark.zip** synthesize the watermark image through the mathematical formula $${I_w}(pi) = \alpha (pi)W(pi) + (1 - \alpha (pi)){I_c}$$ to train the model.

**/datasets/watermark.zip** and **/datasets/test.zip** are used to test image watermark removal models.

Please refer to **def add_watermark_noise(img_train, occupancy=50, self_surpervision=False, same_random=0, alpha=0.3): in utils.py** for the code of synthesizing watermark images.
