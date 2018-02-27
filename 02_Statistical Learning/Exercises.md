

![timu1](https://upload-images.jianshu.io/upload_images/7567244-27af4133d950ad69.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)
![timu2](https://upload-images.jianshu.io/upload_images/7567244-b4d444304aa167aa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)
![timu3](https://upload-images.jianshu.io/upload_images/7567244-bc3bce25addf1012.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)
> 代码如下
+ 计算欧式距离，dist函数默认的计算距离的方式就是欧式距离

```R
for (i in 1:dim(a)[1]){print(dist(rbind(a[i,],b)))}
```
