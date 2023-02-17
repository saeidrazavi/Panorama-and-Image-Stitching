# Panorama and Image Stitching

In this project, our goal is to make panorama from first 900 frame of the original video below: 

<img src="https://user-images.githubusercontent.com/67091916/219731232-cf6b7327-8a84-4e96-ab57-907ec46b5838.gif" width="400" height="250"/>

<img src="https://user-images.githubusercontent.com/67091916/219739757-42e3bdae-a54c-4c4b-bff9-26be32ff89cd.jpg" width="800" height="400"/>


## Panorama using five key frames

To produce a panorama using five key frames. we have some challenges :

*  mapping from some frames to frame 450 is difficult because they share very little area. Therefore we need
to perform a two step mapping , using frame (270,630) as a guide frames and then multiply two homographies
to reach the final homography

* for stitching part, we use both dp(dynamic programming) optimization (minimum cut method) and after finding best boarder , use
Laplacian pyramid blending two stitch these frames with the best result 

if we just combine two frames without any optimization we have : 

<img src="https://user-images.githubusercontent.com/67091916/219745546-3507a226-7f9b-4816-a16a-f5259141b87e.jpg" width="400" height="200"/>

now if we apply dp optimization(minimum error cut) we have : 


<table>
  <tr>
    <td>dp optimization</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/67091916/219745258-59b6422d-c083-482f-929a-71c568caaa76.jpg" width="400" height=200></td>
  </tr>
 </table>
 
after detecting the minimum cut , if we combine two frames without any blending , differences between both
sides of boarder is very clear and obvious . so we use Laplacian pyramid with 9 levels to combine frames very
smoothly and naturally . below you can see the differences :


<table>
  <tr>
    <td>Before Laplacian pyramid</td>
     <td>After Laplacian pyramid</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/67091916/219746094-4017ca83-9ce9-4d68-b5f5-04547dd26acb.jpg" width=400 height=200></td>
    <td><img src="https://user-images.githubusercontent.com/67091916/219746125-a834323a-5471-43ec-9609-3d1f4bf490e6.jpg" width=400 height=200></td>
  </tr>
 </table>
 
we do the above procedure for each two frames that are next to each other with following condition :
• we apply dp optimization for three boarders between two frame (vertical boarder , horizontal boarder
that is on the top of page and horizontal boarder that is in the bottom of the page and make the mask
based one the common areas that these three boarder make).below you can see the final result :


<table>
  <tr>
    <td>panorama of whole video</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/67091916/219757623-a77170e0-b23f-47c3-a338-f7ad01df26c0.jpg" width="800" height=350></td>
  </tr>
 </table>

## Create background panorama

in this section we remove moving objects from the video and create a background panorama .to do this, i divided the whole panorama image vertically into 230 small windows . for each small Windows, i iterate all over the frames and if the projection of frame into frame 450 includes that specific small windows, i save that part into the list . after gathering all of the proper frames that includes the small window , we get median from this list and set the small window to that median . the advantages of this method is that it’s very fast (took around 3 minutes to extract background)

<table>
  <tr>
    <td>background panorama</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/67091916/219758629-ea67f16a-c4ab-4900-9cd3-7d36dad3b128.jpg" width="800" height=350></td>
  </tr>
 </table>

## Create background movie

in this section we should map the background panorama to the movie coordinates. For each frame of the
movie, we need to estimate a projection from the panorama to each frame.  

<img src="https://user-images.githubusercontent.com/67091916/219763744-9e7bb35c-18fb-4c16-b9f9-50d543390e5e.gif" width="400" height="250"/>

## Create forground movie

in this section we want to make a video from moving objects .in each frame we calculate the absolute difference (L1 norm) between original frame and background and set the threshold for these differences. if the differences in a pixel is bigger than this threshold , it can be pixel of moving object . To remove background noise we use two morphological transformation . first we use "opening"(erosion followed by dilation) to remove noises on the mask . we use the kernel with size 5 for erosion to remove noises and then we use "closing"(dilation followed by erosion) to make the mask over moving objects bigger (make it more recognizable)
below you can see the result of these procedure :


<table>
  <tr>
    <td>Mask after threshold</td>
     <td>Opening morphology</td>
     <td>Closing morphology</td>
  </tr>
  <tr>
    <td><img src= "https://user-images.githubusercontent.com/67091916/219771003-0ce81241-1ce9-4216-a1d7-d209a0e2a850.jpg" width=400 height=200></td>
    <td><img src="https://user-images.githubusercontent.com/67091916/219771019-33413921-4006-4db6-a8fe-f8e4386ba097.jpg" width=400 height=200></td>
    <td><img src="https://user-images.githubusercontent.com/67091916/219771026-6fe03acf-09c4-40f6-a1e3-9cb484e61438.jpg" width=400 height=200></td>

  </tr>
 </table>
 
* Result 

<table>
  <tr>
    <td>forground video</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/67091916/219787672-067abb47-0bd4-4eed-8ab9-e9f386dc8d2a.gif" width="400" height=200></td>
  </tr>
 </table>

## Remove camera shake

if we assume that camera parameters change smoothly and obtain a temporally smoothed estimate for each camera parameter, we reach shake less video.

to do so, for each element of homography matrix, we use polynomial approximation. to say it more accurately, for example for element(1,1), we have 900 data from 900 homography matrix and we have to fit the function that best approximate the model. in this section, i use `savgol-filter` from `scipy.signal` library and fit the polynomial with 7 degree and kernel size of 351 for these 900data. for better intuition about smoothing noisy data see the picture below :

<table>
  <tr>
    <td>smoothing data</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/67091916/219778183-ca4cdf69-607b-4855-8f94-fb9a564a9cdb.png" width="400" height=200></td>
  </tr>
 </table>

* Result 


<table>
  <tr>
    <td>smoothed video</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/67091916/219787745-b0949aa9-524d-4384-a58e-23fc18fcac5b.gif" width="400" height=200></td>
  </tr>
 </table>

